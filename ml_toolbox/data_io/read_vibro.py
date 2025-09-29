"""
Read vibration signal data from binary files.
"""

import numpy as np
from pathlib import Path
from typing import Union
from .read_raw import read_raw


def read_vibro(file_path: Union[str, Path]) -> np.ndarray:
    """
    Read binary ADC vibration signal.
    
    This function reads vibration sensor data from binary files.
    Vibration data typically has 4 channels.
    
    Args:
        file_path: Path to binary file
    
    Returns:
        np.ndarray: Raw data with shape [samples, channels]
    """
    # Read raw data using base function (4 channels for vibration)
    data = read_raw(file_path, 4)
    return data
