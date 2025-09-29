"""
Read raw binary ADC data from files.
"""

import numpy as np
from pathlib import Path
from typing import Union


def read_raw(file_path: Union[str, Path], num_channels: int) -> np.ndarray:
    """
    Read interleaved multi-channel ADC data from binary file.
    
    Args:
        file_path: Path to the binary file
        num_channels: Number of channels in the file (e.g. 2 or 4)
    
    Returns:
        np.ndarray: Data array with shape [samples, channels]
        
    Raises:
        FileNotFoundError: If the file cannot be opened
        ValueError: If the file format is invalid
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Could not open file: {file_path}")
    
    try:
        # Read as 64-bit floats (equivalent to MATLAB's 'double')
        raw_data = np.fromfile(file_path, dtype=np.float64)
        
        # Calculate total samples and trim excess data if needed
        total_samples = len(raw_data) // num_channels
        raw_data = raw_data[:total_samples * num_channels]
        
        # Reshape to [samples, channels] - transpose equivalent to MATLAB
        data = raw_data.reshape(num_channels, total_samples).T
        
        return data
        
    except Exception as e:
        raise ValueError(f"Error reading file {file_path}: {str(e)}")
