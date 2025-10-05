"""
Read and preprocess current signal data from binary files.
"""

import numpy as np
from pathlib import Path
from typing import Union
from scipy.signal import medfilt, butter, filtfilt
from .read_raw import read_raw


def read_current(
    file_path: Union[str, Path],
    apply_filter: bool = False,
    sampling_freq: float = 10000,
    cutoff_freq: float = 3500,
    median_window: int = 7,
) -> np.ndarray:
    """
    Read binary ADC current signal with preprocessing.
    
    This function reads current sensor data and applies:
    1. DC offset removal (mean subtraction)
    2. Optional: Median filtering for despiking + Butterworth lowpass filter
    
    Args:
        file_path: Path to binary file
        apply_filter: If True, apply median filter + Butterworth lowpass filter.
            Default is False (no filtering beyond DC offset removal).
        sampling_freq: Sampling frequency in Hz. Default is 10000 Hz.
        cutoff_freq: Lowpass filter cutoff frequency in Hz. Default is 3500 Hz.
        median_window: Median filter window size for despiking. Default is 7.
    
    Returns:
        np.ndarray: Processed data with shape [samples, channels] and fixed ADC offset removed.
            If apply_filter is False, this is just DC-offset-removed data.
            If apply_filter is True, applies median despiking + Butterworth lowpass filter.
    """
    # ADC_OFFSET = 2.5  # Original fixed offset
  
    # Read raw data using base function (2 channels for current)
    raw_data = read_raw(file_path, 2)
    
    # Remove DC offset by subtracting mean
    data = raw_data - np.mean(raw_data, axis=0)
    
    if not apply_filter:
        # By default, return only DC-offset-removed data (no filtering)
        return data

    # Apply median filter for despiking + Butterworth lowpass filter
    filtered_data = np.zeros_like(data)
    
    # Design Butterworth 4th order lowpass filter
    nyquist = sampling_freq / 2
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    
    for channel in range(data.shape[1]):
        # Step 1: Apply median filter for despiking
        despiked = medfilt(data[:, channel], kernel_size=median_window)
        
        # Step 2: Apply Butterworth lowpass filter with zero-phase filtering
        filtered_data[:, channel] = filtfilt(b, a, despiked)

    return filtered_data
