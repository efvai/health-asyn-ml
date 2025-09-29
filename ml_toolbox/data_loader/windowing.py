"""
Windowing module for time series data preprocessing.

This module provides utilities for splitting time series data into windows
for feature extraction and machine learning applications.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Generator
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class WindowConfig:
    """Configuration for windowing parameters."""
    window_size: int
    step_size: int
    overlap_ratio: Optional[float] = None
    padding: bool = False
    min_window_size: Optional[int] = None
    
    def __post_init__(self):
        """Validate and compute derived parameters."""
        if self.overlap_ratio is not None:
            if not 0 <= self.overlap_ratio < 1:
                raise ValueError("overlap_ratio must be between 0 and 1 (exclusive)")
            self.step_size = int(self.window_size * (1 - self.overlap_ratio))
        
        if self.min_window_size is None:
            self.min_window_size = self.window_size
        
        if self.step_size <= 0:
            raise ValueError("step_size must be positive")


class WindowExtractor:
    """Extract windows from time series data with various strategies."""
    
    def __init__(self, config: WindowConfig):
        self.config = config
    
    def extract_windows(self, 
                       data: np.ndarray, 
                       metadata: Optional[Dict] = None) -> Tuple[np.ndarray, List[Dict]]:
        """
        Extract windows from time series data.
        
        Args:
            data: Input data array with shape (samples, channels)
            metadata: Optional metadata dictionary to propagate to windows
            
        Returns:
            Tuple of (windowed_data, window_metadata_list)
            windowed_data shape: (n_windows, window_size, channels)
        """
        if len(data.shape) != 2:
            raise ValueError("Input data must be 2D (samples, channels)")
        
        n_samples, n_channels = data.shape
        windows = []
        window_metadata = []
        
        # Calculate window positions
        start_positions = range(0, n_samples - self.config.window_size + 1, self.config.step_size)
        
        for i, start_pos in enumerate(start_positions):
            end_pos = start_pos + self.config.window_size
            
            # Extract window
            window = data[start_pos:end_pos, :]
            
            # Only add if window meets minimum size requirement
            if window.shape[0] >= self.config.min_window_size:
                windows.append(window)
                
                # Create metadata for this window
                win_meta = {
                    'window_id': i,
                    'start_sample': start_pos,
                    'end_sample': end_pos,
                    'window_size': window.shape[0],
                    'n_channels': n_channels
                }
                
                # Propagate original metadata
                if metadata:
                    win_meta.update(metadata)
                    
                window_metadata.append(win_meta)
        
        if not windows:
            logger.warning(f"No windows extracted from data with {n_samples} samples")
            return np.array([]), []
        
        # Handle padding if requested and last window is smaller
        if self.config.padding and windows:
            last_window = windows[-1]
            if last_window.shape[0] < self.config.window_size:
                # Pad with zeros
                padding_size = self.config.window_size - last_window.shape[0]
                padding = np.zeros((padding_size, n_channels))
                windows[-1] = np.vstack([last_window, padding])
                window_metadata[-1]['padded'] = True
                window_metadata[-1]['padding_size'] = padding_size
        
        return np.array(windows), window_metadata
    
    def extract_windows_batch(self, 
                             data_list: List[np.ndarray], 
                             metadata_list: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        """
        Extract windows from multiple time series files.
        
        Args:
            data_list: List of data arrays
            metadata_list: List of metadata dictionaries
            
        Returns:
            Tuple of (all_windows, all_window_metadata)
        """
        all_windows = []
        all_metadata = []
        
        for data, metadata in zip(data_list, metadata_list):
            windows, win_metadata = self.extract_windows(data, metadata)
            
            if len(windows) > 0:
                all_windows.extend(windows)
                all_metadata.extend(win_metadata)
        
        if not all_windows:
            return np.array([]), []
        
        return np.array(all_windows), all_metadata


class StratifiedWindowExtractor(WindowExtractor):
    """Window extractor with stratified sampling to balance classes."""
    
    def extract_stratified_windows(self, 
                                  data_list: List[np.ndarray], 
                                  metadata_list: List[Dict],
                                  target_key: str = 'condition',
                                  max_windows_per_class: Optional[int] = None) -> Tuple[np.ndarray, List[Dict]]:
        """
        Extract windows with stratified sampling to balance classes.
        
        Args:
            data_list: List of data arrays
            metadata_list: List of metadata dictionaries
            target_key: Key in metadata that defines the class/target
            max_windows_per_class: Maximum windows per class (None for no limit)
            
        Returns:
            Tuple of (balanced_windows, balanced_metadata)
        """
        # Group by class
        class_data = {}
        
        for data, metadata in zip(data_list, metadata_list):
            class_label = metadata.get(target_key, 'unknown')
            
            if class_label not in class_data:
                class_data[class_label] = {'data': [], 'metadata': []}
            
            class_data[class_label]['data'].append(data)
            class_data[class_label]['metadata'].append(metadata)
        
        # Extract windows for each class
        balanced_windows = []
        balanced_metadata = []
        
        for class_label, class_info in class_data.items():
            windows, win_metadata = self.extract_windows_batch(
                class_info['data'], 
                class_info['metadata']
            )
            
            if len(windows) > 0:
                # Apply limit if specified
                if max_windows_per_class and len(windows) > max_windows_per_class:
                    # Random sampling to get desired number
                    indices = np.random.choice(len(windows), max_windows_per_class, replace=False)
                    windows = windows[indices]
                    win_metadata = [win_metadata[i] for i in indices]
                
                balanced_windows.extend(windows)
                balanced_metadata.extend(win_metadata)
                
                logger.info(f"Class '{class_label}': extracted {len(windows)} windows")
        
        if not balanced_windows:
            return np.array([]), []
        
        return np.array(balanced_windows), balanced_metadata


class SlidingWindowGenerator:
    """Generator for memory-efficient sliding window processing."""
    
    def __init__(self, config: WindowConfig):
        self.config = config
    
    def generate_windows(self, 
                        data: np.ndarray, 
                        metadata: Optional[Dict] = None) -> Generator[Tuple[np.ndarray, Dict], None, None]:
        """
        Generate windows one at a time for memory efficiency.
        
        Args:
            data: Input data array with shape (samples, channels)
            metadata: Optional metadata dictionary
            
        Yields:
            Tuple of (window, window_metadata)
        """
        if len(data.shape) != 2:
            raise ValueError("Input data must be 2D (samples, channels)")
        
        n_samples, n_channels = data.shape
        start_positions = range(0, n_samples - self.config.window_size + 1, self.config.step_size)
        
        for i, start_pos in enumerate(start_positions):
            end_pos = start_pos + self.config.window_size
            window = data[start_pos:end_pos, :]
            
            if window.shape[0] >= self.config.min_window_size:
                win_meta = {
                    'window_id': i,
                    'start_sample': start_pos,
                    'end_sample': end_pos,
                    'window_size': window.shape[0],
                    'n_channels': n_channels
                }
                
                if metadata:
                    win_meta.update(metadata)
                
                yield window, win_meta


class WindowAnalyzer:
    """Analyze windowing characteristics and statistics."""
    
    @staticmethod
    def analyze_windowing(data_length: int, config: WindowConfig) -> Dict:
        """
        Analyze windowing parameters for given data length.
        
        Args:
            data_length: Length of the input data
            config: Window configuration
            
        Returns:
            Dictionary with windowing analysis
        """
        if data_length < config.window_size:
            return {
                'feasible': False,
                'reason': f'Data length ({data_length}) < window size ({config.window_size})',
                'n_windows': 0,
                'coverage_ratio': 0.0
            }
        
        n_windows = (data_length - config.window_size) // config.step_size + 1
        last_window_start = (n_windows - 1) * config.step_size
        last_window_end = last_window_start + config.window_size
        coverage_ratio = last_window_end / data_length
        
        overlap_samples = config.window_size - config.step_size
        overlap_ratio = overlap_samples / config.window_size if config.window_size > 0 else 0
        
        return {
            'feasible': True,
            'n_windows': n_windows,
            'coverage_ratio': coverage_ratio,
            'overlap_samples': overlap_samples,
            'overlap_ratio': overlap_ratio,
            'unused_samples': data_length - last_window_end,
            'total_samples_in_windows': n_windows * config.window_size,
            'data_expansion_ratio': (n_windows * config.window_size) / data_length
        }
    
    @staticmethod
    def suggest_window_config(data_length: int, 
                            target_windows: int, 
                            overlap_ratio: float = 0.5) -> WindowConfig:
        """
        Suggest window configuration for target number of windows.
        
        Args:
            data_length: Length of the input data
            target_windows: Desired number of windows
            overlap_ratio: Desired overlap ratio
            
        Returns:
            Suggested WindowConfig
        """
        # Calculate window size to achieve target number of windows
        # n_windows = (data_length - window_size) / step_size + 1
        # step_size = window_size * (1 - overlap_ratio)
        # Solving for window_size:
        
        step_ratio = 1 - overlap_ratio
        window_size = int((data_length + target_windows - 1) / (target_windows * step_ratio + (1 - step_ratio)))
        
        # Ensure reasonable bounds
        window_size = max(window_size, 32)  # Minimum 32 samples
        window_size = min(window_size, data_length // 2)  # Maximum half the data
        
        return WindowConfig(
            window_size=window_size,
            step_size=int(window_size * step_ratio),
            overlap_ratio=overlap_ratio
        )


def create_windows_for_ml(data_list: List[np.ndarray], 
                         metadata_list: List[Dict],
                         window_size: int,
                         overlap_ratio: float = 0.5,
                         balance_classes: bool = True,
                         max_windows_per_class: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Convenience function to create windows ready for ML training.
    
    Args:
        data_list: List of data arrays
        metadata_list: List of metadata dictionaries
        window_size: Size of each window
        overlap_ratio: Overlap between consecutive windows
        balance_classes: Whether to balance classes
        max_windows_per_class: Maximum windows per class
        
    Returns:
        Tuple of (X, y, window_metadata)
        X: Window data (n_windows, window_size, n_channels)
        y: Labels (n_windows,)
        window_metadata: List of window metadata
    """
    config = WindowConfig(
        window_size=window_size,
        step_size=int(window_size * (1 - overlap_ratio)),
        overlap_ratio=overlap_ratio
    )
    
    if balance_classes:
        extractor = StratifiedWindowExtractor(config)
        windows, win_metadata = extractor.extract_stratified_windows(
            data_list, metadata_list, 
            max_windows_per_class=max_windows_per_class
        )
    else:
        extractor = WindowExtractor(config)
        windows, win_metadata = extractor.extract_windows_batch(data_list, metadata_list)
    
    if len(windows) == 0:
        return np.array([]), np.array([]), []
    
    # Extract labels (assuming condition mapping exists)
    condition_map = {
        "healthy": 0,
        "faulty_bearing": 1, 
        "misalignment": 2,
        "system_misalignment": 3
    }
    
    labels = []
    for meta in win_metadata:
        condition = meta.get('condition', 'unknown')
        label = condition_map.get(condition, -1)
        labels.append(label)
    
    return windows, np.array(labels), win_metadata