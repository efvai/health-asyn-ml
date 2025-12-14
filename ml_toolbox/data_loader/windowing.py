"""
Windowing module for time series data preprocessing.

This module provides utilities for splitting time series data into windows
for feature extraction and machine learning applications.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)
# Prevent messages from being propagated to the root logger (avoids duplicate output
# in environments that configure logging handlers, e.g. notebooks)
#logger.propagate = False

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
                                  max_windows_per_class: Optional[int] = None,
                                  random_state: Optional[int] = None,
                                  shuffle: bool = True) -> Tuple[np.ndarray, List[Dict]]:
        """Extract windows with stratified sampling to balance classes.

        Args:
            data_list: Raw signals grouped by file.
            metadata_list: Metadata dictionaries aligned to ``data_list``.
            target_key: Metadata key holding the class/condition label.
            max_windows_per_class: Optional cap per class after balancing.
            random_state: Seed for deterministic sampling.
            shuffle: Whether to shuffle the aggregated balanced windows.

        Returns:
            Balanced window tensor and associated metadata list.
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
        rng = np.random.RandomState(random_state) if random_state is not None else None
        class_windows: Dict[str, np.ndarray] = {}
        class_metadata: Dict[str, List[Dict]] = {}
        dropped_classes: List[str] = []

        for class_label, class_info in class_data.items():
            windows, win_metadata = self.extract_windows_batch(
                class_info['data'],
                class_info['metadata']
            )

            if len(windows) == 0:
                dropped_classes.append(class_label)
                continue

            class_windows[class_label] = windows
            class_metadata[class_label] = win_metadata

        if dropped_classes:
            logger.warning(
                "Classes without extracted windows will be dropped: %s",
                ", ".join(sorted(dropped_classes))
            )

        if not class_windows:
            return np.array([]), []

        class_counts = {label: windows.shape[0] for label, windows in class_windows.items()}
        min_class_count = min(class_counts.values())

        if max_windows_per_class is not None:
            if max_windows_per_class <= 0:
                raise ValueError("max_windows_per_class must be positive when provided")
            per_class_n = min(max_windows_per_class, min_class_count)
        else:
            per_class_n = min_class_count

        if per_class_n == 0:
            logger.warning(
                "No windows available after balancing (per_class_n computed as 0)"
            )
            return np.array([]), []

        balanced_windows: List[np.ndarray] = []
        balanced_metadata: List[Dict] = []

        for class_label, windows in class_windows.items():
            metadata = class_metadata[class_label]

            if windows.shape[0] > per_class_n:
                if rng is not None:
                    indices = rng.choice(windows.shape[0], per_class_n, replace=False)
                else:
                    indices = np.random.choice(windows.shape[0], per_class_n, replace=False)
            else:
                indices = np.arange(windows.shape[0])

            for idx in indices:
                balanced_windows.append(windows[idx])
                balanced_metadata.append(metadata[idx])

            logger.info(
                "Class '%s': selected %d of %d available windows",
                class_label,
                len(indices),
                windows.shape[0]
            )

        if not balanced_windows:
            return np.array([]), []

        if shuffle and len(balanced_windows) > 1:
            if rng is not None:
                perm = rng.permutation(len(balanced_windows))
            else:
                perm = np.random.permutation(len(balanced_windows))
            balanced_windows = [balanced_windows[i] for i in perm]
            balanced_metadata = [balanced_metadata[i] for i in perm]

        return np.stack(balanced_windows), balanced_metadata

def create_windows_for_ml(data_list: List[np.ndarray], 
                         metadata_list: List[Dict],
                         window_size: int,
                         overlap_ratio: float = 0.5,
                         max_windows_per_class: Optional[int] = None,
                         *,
                         target_key: str = 'condition',
                         condition_map: Optional[Dict[str, int]] = None,
                         random_state: Optional[int] = None,
                         shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Convenience function to create windows ready for ML training.
    
    Args:
        data_list: List of data arrays
        metadata_list: List of metadata dictionaries
        window_size: Size of each window
        overlap_ratio: Overlap between consecutive windows
        max_windows_per_class: Maximum windows per class
        target_key: Metadata key that holds the class label
        condition_map: Optional mapping from condition names to numeric labels
        unknown_label: Label assigned when condition is missing from the map
        random_state: Seed for deterministic sampling
        shuffle: Whether to shuffle the balanced window set
        
    Returns:
        Tuple of (X, y, window_metadata)
        X: Window data (n_windows, window_size, n_channels)
        y: Labels (n_windows,)
        window_metadata: List of window metadata
    """
    step_size = int(window_size * (1 - overlap_ratio))
    if step_size <= 0:
        raise ValueError("overlap_ratio results in non-positive step size; choose smaller overlap")

    config = WindowConfig(
        window_size=window_size,
        step_size=step_size,
        overlap_ratio=overlap_ratio
    )
    
    extractor = StratifiedWindowExtractor(config)
    windows, win_metadata = extractor.extract_stratified_windows(
        data_list,
        metadata_list,
        target_key=target_key,
        max_windows_per_class=max_windows_per_class,
        random_state=random_state,
        shuffle=shuffle
    )
    
    if len(windows) == 0:
        return np.array([]), np.array([]), []
    
    if condition_map is None:
        observed_conditions = {
            meta.get(target_key, 'unknown')
            for meta in win_metadata
            if meta.get(target_key, 'unknown') != 'unknown'
        }
        condition_map = {
            condition: idx for idx, condition in enumerate(sorted(observed_conditions))
        }

    labels = []
    for meta in win_metadata:
        condition = meta.get(target_key, 'unknown')
        label = condition_map.get(condition, -1)
        labels.append(label)

    return windows, np.array(labels, dtype=np.int32), win_metadata