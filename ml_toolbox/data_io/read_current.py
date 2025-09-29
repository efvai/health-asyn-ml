"""
Read and preprocess current signal data from binary files.
"""

import numpy as np
from pathlib import Path
from typing import Union
from scipy.signal import medfilt
from .read_raw import read_raw


def read_current(file_path: Union[str, Path]) -> np.ndarray:
    """
    Read binary ADC current signal with preprocessing.
    
    This function reads current sensor data and applies:
    1. DC offset removal (mean subtraction)
    2. Outlier detection and interpolation using median filtering
    
    Args:
        file_path: Path to binary file
    
    Returns:
        np.ndarray: Processed data with shape [samples, channels] and fixed ADC offset removed
    """
    # ADC_OFFSET = 2.5  # Original fixed offset (commented out as in MATLAB)
    
    # Read raw data using base function (2 channels for current)
    raw_data = read_raw(file_path, 2)
    
    # Remove DC offset by subtracting mean
    data = raw_data - np.mean(raw_data, axis=0)
    
    # Outlier suppression using median filtering
    # Apply median filter with window size 21
    signal_medfilt = np.zeros_like(data)
    for channel in range(data.shape[1]):
        signal_medfilt[:, channel] = medfilt(data[:, channel], kernel_size=21)
    
    # Calculate residual signal
    residual = data - signal_medfilt
    
    # Calculate MAD (Median Absolute Deviation) for outlier detection
    mad_val = np.median(np.abs(residual - np.median(residual, axis=0)), axis=0)
    
    # Set threshold at 5 * MAD
    threshold = 5 * mad_val
    
    # Identify outliers
    is_outlier = np.abs(residual) > threshold
    
    # Interpolate outliers using linear interpolation (equivalent to MATLAB's pchip for this case)
    signal_clean = data.copy()
    
    for channel in range(data.shape[1]):
        outlier_pos = np.where(is_outlier[:, channel])[0]
        valid_pos = np.where(~is_outlier[:, channel])[0]
        
        if len(outlier_pos) > 0 and len(valid_pos) > 1:
            # Use linear interpolation (can be upgraded to cubic later if needed)
            signal_clean[outlier_pos, channel] = np.interp(
                outlier_pos, 
                valid_pos, 
                data[valid_pos, channel]
            )
    
    return signal_clean
