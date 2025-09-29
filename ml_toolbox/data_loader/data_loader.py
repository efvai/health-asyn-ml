"""
Efficient data loading with caching and preprocessing pipelines.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
import logging
from .dataset_manager import DatasetManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Efficient data loader with caching and parallel processing."""
    
    def __init__(self, dataset_path: Path):
        self.dataset_manager = DatasetManager(dataset_path)
        self._index = None
    
    @property
    def index(self):
        """Lazy load dataset index."""
        if self._index is None:
            self._index = self.dataset_manager.get_index()
        return self._index
    
    def load_batch(self, 
                   condition: Optional[str] = None,
                   load: Optional[str] = None, 
                   frequency: Optional[str] = None,
                   sensor_type: Optional[str] = None,
                   max_workers: int = 4) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Load batch of data with optional filtering.
        
        Args:
            condition: Filter by condition (e.g., 'healthy', 'faulty_bearing')
            load: Filter by load (e.g., 'no_load', 'under_load')
            frequency: Filter by frequency (e.g., '10hz_1', '20hz_2')
            sensor_type: Filter by sensor type ('current', 'vibration')
            max_workers: Number of parallel workers for loading
            
        Returns:
            Tuple of (data_list, metadata_list)
        """
        
        # Filter files based on criteria
        filtered_files = self.dataset_manager.filter_files(condition, load, frequency, sensor_type)
        
        if not filtered_files:
            logger.warning(f"No files found matching criteria: condition={condition}, load={load}, frequency={frequency}, sensor_type={sensor_type}")
            return [], []
        
        logger.info(f"Loading {len(filtered_files)} files with {max_workers} workers")
        
        # Load data in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            data_futures = [
                    executor.submit(self.dataset_manager.load_sample, file_info) 
                    for file_info in filtered_files
            ]
            
            data_list = []
            metadata_list = []
            
            for future, file_info in zip(data_futures, filtered_files):
                try:
                    data = future.result()
                    data_list.append(data)
                    metadata_list.append(file_info)
                except Exception as e:
                    logger.error(f"Error loading {file_info['path']}: {e}")
        
        logger.info(f"Successfully loaded {len(data_list)} files")
        return data_list, metadata_list
    
    def load_batch_as_arrays(self, 
                           condition: Optional[str] = None,
                           load: Optional[str] = None, 
                           frequency: Optional[str] = None,
                           sensor_type: Optional[str] = None,
                           max_workers: int = 4) -> Tuple:
        """
        Load batch of data and return as numpy arrays suitable for ML.
        
        Returns:
            Tuple of (data_array, labels_array, metadata_list)
        """
        data_list, metadata_list = self.load_batch(
            condition, load, frequency, sensor_type, max_workers
        )
        
        if not data_list:
            return np.array([]), np.array([]), []
        
        # Convert to arrays
        try:
            # Stack data (assuming all have same shape)
            data_array = np.stack(data_list, axis=0)
            labels_array = np.array([self._encode_label(meta) for meta in metadata_list])
            
            return data_array, labels_array, metadata_list
        except ValueError as e:
            logger.error(f"Error stacking data arrays: {e}")
            # Return as lists if stacking fails (different shapes)
            return data_list, np.array([self._encode_label(meta) for meta in metadata_list]), metadata_list
    
    def _encode_label(self, file_info: Dict) -> int:
        """Encode file info as numerical label."""
        condition_map = {
            "healthy": 0,
            "faulty_bearing": 1, 
            "misalignment": 2,
            "system_misalignment": 3
        }
        return condition_map.get(file_info["condition"], -1)
    
    def get_label_mapping(self) -> Dict[int, str]:
        """Get mapping from numerical labels to condition names."""
        return {
            0: "healthy",
            1: "faulty_bearing", 
            2: "misalignment",
            3: "system_misalignment"
        }
    
    def load_pairs(self, 
                   condition: Optional[str] = None,
                   load: Optional[str] = None, 
                   frequency: Optional[str] = None,
                   max_workers: int = 4) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict]]:
        """
        Load paired current and vibration data from the same measurement.
        
        Returns:
            Tuple of (current_data_list, vibration_data_list, metadata_list)
        """
        
        # Get current and vibration files separately
        current_files = self.dataset_manager.filter_files(condition, load, frequency, "current")
        vibration_files = self.dataset_manager.filter_files(condition, load, frequency, "vibration")
        
        # Create a mapping for pairing based on condition, load, and frequency
        current_map = {}
        for file_info in current_files:
            key = (file_info["condition"], file_info["load"], file_info["frequency"])
            current_map[key] = file_info
        
        vibration_map = {}
        for file_info in vibration_files:
            key = (file_info["condition"], file_info["load"], file_info["frequency"])
            vibration_map[key] = file_info
        
        # Find common keys (paired measurements)
        common_keys = set(current_map.keys()) & set(vibration_map.keys())
        
        if not common_keys:
            logger.warning("No paired current and vibration files found")
            return [], [], []
        
        logger.info(f"Found {len(common_keys)} paired measurements")
        
        # Load paired data
        current_data_list = []
        vibration_data_list = []
        metadata_list = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for key in common_keys:
                current_file = current_map[key]
                vibration_file = vibration_map[key]

                current_future = executor.submit(self.dataset_manager.load_sample, current_file)
                vibration_future = executor.submit(self.dataset_manager.load_sample, vibration_file)
                
                try:
                    current_data = current_future.result()
                    vibration_data = vibration_future.result()
                    
                    current_data_list.append(current_data)
                    vibration_data_list.append(vibration_data)
                    
                    # Use current file metadata as reference
                    paired_metadata = current_file.copy()
                    paired_metadata["paired_vibration_file"] = vibration_file["path"]
                    metadata_list.append(paired_metadata)
                    
                except Exception as e:
                    logger.error(f"Error loading paired data for {key}: {e}")
        
        logger.info(f"Successfully loaded {len(current_data_list)} paired measurements")
        return current_data_list, vibration_data_list, metadata_list