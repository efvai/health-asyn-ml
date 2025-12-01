"""
Efficient data loading pipeline.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Sequence, Union
from concurrent.futures import ThreadPoolExecutor
import logging
from .dataset_manager import DatasetManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
#logger.propagate = False


class DataLoader:
    """Efficient data loader with concurrent processing."""
    
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
                   max_workers: int = 4,
                   *,
                   conditions: Optional[Union[str, Sequence[str]]] = None,
                   loads: Optional[Union[str, Sequence[str]]] = None,
                   frequencies: Optional[Union[str, Sequence[str]]] = None,
                   frequency_dirs: Optional[Union[str, Sequence[str]]] = None,
                   sensor_types: Optional[Union[str, Sequence[str]]] = None) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Load batch of data with optional filtering.
        
        Args:
            condition: Filter by a single condition (e.g., 'healthy').
            load: Filter by a single load level (e.g., 'no_load').
            frequency: Filter by a base frequency (e.g., '10hz').
            sensor_type: Filter by a single sensor type ('current', 'vibration').
            max_workers: Number of parallel workers for loading.
            conditions: One or more conditions to include. Supersedes ``condition`` when provided.
            loads: One or more load levels to include. Supersedes ``load`` when provided.
            frequencies: One or more base frequencies to include. Supersedes ``frequency`` when provided.
            frequency_dirs: One or more specific frequency directories (e.g., '10hz_1').
            sensor_types: One or more sensor types to include. Supersedes ``sensor_type`` when provided.
            
        Returns:
            Tuple of (data_list, metadata_list)
        """
        
        # Filter files based on criteria
        filtered_files = self.dataset_manager.filter_files(
            condition=condition,
            load=load,
            frequency=frequency,
            sensor_type=sensor_type,
            conditions=conditions,
            loads=loads,
            frequencies=frequencies,
            frequency_dirs=frequency_dirs,
            sensor_types=sensor_types,
        )
        
        if not filtered_files:
            logger.warning(
                "No files found matching criteria: "
                f"condition={condition or conditions}, "
                f"load={load or loads}, "
                f"frequency={frequency or frequencies}, "
                f"frequency_dir={frequency_dirs}, "
                f"sensor_type={sensor_type or sensor_types}"
            )
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
    
    def get_label_mapping(self) -> Dict[int, str]:
        """Get mapping from numerical labels to condition names."""
        return {
            0: "healthy",
            1: "faulty_bearing", 
            2: "misalignment",
            3: "system_misalignment"
        }