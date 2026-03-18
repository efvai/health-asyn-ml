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
                   max_workers: int = 4,
                   *,
                   classes: Optional[Union[str, Sequence[str]]] = None,
                   loads: Optional[Union[float, int, Sequence[Union[float, int]]]] = None,
                   frequencies: Optional[Union[float, int, Sequence[Union[float, int]]]] = None,
                   sensor_types: Optional[Union[str, Sequence[str]]] = None,
                   sample_ids: Optional[Union[str, Sequence[str]]] = None) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Load batch of data with optional filtering.
        
        Args:
            max_workers: Number of parallel workers for loading.
            classes: One or more class labels to include.
            loads: One or more numeric load values to include.
            frequencies: One or more electrical frequencies (Hz) to include.
            sensor_types: One or more sensor types to include.
            sample_ids: Optional sample directory names to include (e.g., '0001').
            
        Returns:
            Tuple of (data_list, metadata_list)
        """
        
        # Filter files based on criteria
        filtered_files = self.dataset_manager.filter_files(
            classes=classes,
            loads=loads,
            frequencies=frequencies,
            sensor_types=sensor_types,
            sample_ids=sample_ids,
        )
        
        if not filtered_files:
            logger.warning(
                "No files found matching criteria: "
                f"classes={classes}, "
                f"loads={loads}, "
                f"frequencies={frequencies}, "
                f"sensor_types={sensor_types}"
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
    