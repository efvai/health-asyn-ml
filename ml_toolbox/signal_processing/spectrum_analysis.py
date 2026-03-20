"""
Spectrum analysis utilities for signal processing.

This module provides functions for frequency domain analysis of signals,
including FFT computation and spectral peak detection.
"""

import numpy as np
from scipy.signal import welch, find_peaks
from typing import Dict, Optional, Any


def prepare_time_frequency_view(sample: np.ndarray,
                                metadata: Dict[str, Any],
                                channel_index: int = 0,
                                nperseg: Optional[int] = None,
                                normalize: bool = True) -> Dict[str, Any]:
    """
    Prepare time-domain and frequency-domain data for one channel of a sample.

    Parameters
    ----------
    sample : np.ndarray
        Loaded sample array with shape ``[samples, channels]``.
    metadata : Dict[str, Any]
        Metadata for the sample. Must contain ``sensor_type`` and the matching
        sampling-rate field for that sensor.
    channel_index : int, default=0
        Zero-based channel index to extract.
    nperseg : int, optional
        Segment length forwarded to ``compute_fft_spectrum``.
    normalize : bool, default=True
        Whether to normalize the spectrum.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing ``channel_signal``, ``time_axis``,
        ``sampling_rate`` and ``spectrum``.
    """
    sample_array = np.asarray(sample)
    if sample_array.ndim != 2:
        raise ValueError("sample must have shape [samples, channels]")

    if channel_index < 0 or channel_index >= sample_array.shape[1]:
        raise IndexError(
            f"channel_index {channel_index} is out of range for {sample_array.shape[1]} channels"
        )

    sensor_type = metadata.get("sensor_type")
    if sensor_type == "vibration":
        sampling_rate_key = "sample_rate_vibro_hz"
    elif sensor_type == "current":
        sampling_rate_key = "sample_rate_current_hz"
    else:
        raise KeyError("sensor_type")

    if sampling_rate_key not in metadata:
        raise KeyError(sampling_rate_key)

    sampling_rate = float(metadata[sampling_rate_key])
    channel_signal = sample_array[:, channel_index]
    time_axis = np.arange(channel_signal.shape[0], dtype=float) / sampling_rate
    spectrum = compute_fft_spectrum(
        channel_signal,
        sampling_rate=sampling_rate,
        nperseg=nperseg,
        normalize=normalize,
    )

    return {
        "channel_signal": channel_signal,
        "time_axis": time_axis,
        "sampling_rate": sampling_rate,
        "spectrum": spectrum,
    }


def compute_fft_spectrum(signal: np.ndarray, 
                        sampling_rate: float,
                        nperseg: Optional[int] = None,
                        normalize: bool = True) -> Dict[str, Any]:
    """
    Compute FFT spectrum of a signal.
    
    This is a general-purpose function for computing frequency spectra
    of any signal, not tied to envelope analysis.
    
    Parameters:
    -----------
    signal : np.ndarray
        Input signal to analyze
    sampling_rate : float
        Sampling rate of the signal in Hz
    nperseg : int, optional
        Length of each segment for Welch's method. If None, uses standard FFT.
        Welch's method provides better noise reduction for longer signals.
    normalize : bool, default=True
        Whether to normalize the magnitude spectrum to [0, 1]
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing:
        - 'freqs': np.ndarray - Frequency array in Hz
        - 'magnitude': np.ndarray - Magnitude spectrum
        - 'power': np.ndarray - Power spectrum (magnitude squared)
        - 'sampling_rate': float - Sampling rate used
        
    Examples:
    ---------
    >>> signal = np.sin(2 * np.pi * 50 * np.linspace(0, 1, 1000))
    >>> spectrum = compute_fft_spectrum(signal, sampling_rate=1000.0)
    >>> peak_freq = spectrum['freqs'][np.argmax(spectrum['magnitude'])]
    >>> print(f"Peak frequency: {peak_freq:.1f} Hz")
    """
    if nperseg is not None:
        # Use Welch's method for better noise reduction
        freqs, psd = welch(signal, fs=sampling_rate, nperseg=nperseg, 
                          scaling='density', detrend='constant')
        magnitude = np.sqrt(psd * sampling_rate / 2)  # Convert PSD to magnitude
        power = psd
    else:
        # Use standard FFT
        fft_vals = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/sampling_rate)
        
        # Take only positive frequencies
        pos_mask = freqs >= 0
        freqs = freqs[pos_mask]
        fft_vals = fft_vals[pos_mask]
        
        magnitude = np.abs(fft_vals) / len(signal)
        power = magnitude ** 2
    
    # Normalize if requested
    if normalize:
        magnitude = magnitude / np.max(magnitude) if np.max(magnitude) != 0 else magnitude
        power = power / np.max(power) if np.max(power) != 0 else power
    
    return {
        'freqs': freqs,
        'magnitude': magnitude,
        'power': power,
        'sampling_rate': sampling_rate
    }


def find_spectral_peaks(spectrum_data: Dict[str, Any],
                       height: Optional[float] = None,
                       prominence: Optional[float] = None,
                       distance: Optional[int] = None,
                       num_peaks: int = 10) -> Dict[str, np.ndarray]:
    """
    Find peaks in a frequency spectrum.
    
    This function identifies prominent frequencies in a spectrum computed
    by compute_fft_spectrum() or similar functions.
    
    Parameters:
    -----------
    spectrum_data : Dict[str, Any]
        Spectrum dictionary containing 'freqs' and 'magnitude' arrays.
        Typically output from compute_fft_spectrum().
    height : float, optional
        Minimum height of peaks relative to the spectrum baseline.
        If None, no height filtering is applied.
    prominence : float, optional
        Minimum prominence of peaks. Prominence measures how much a peak
        stands out relative to surrounding valleys.
    distance : int, optional
        Minimum distance between peaks in frequency samples.
        Prevents detection of very close peaks.
    num_peaks : int, default=10
        Maximum number of peaks to return. Returns the highest magnitude peaks.
        
    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary containing:
        - 'peak_freqs': np.ndarray - Frequencies of detected peaks
        - 'peak_magnitudes': np.ndarray - Magnitudes at peak frequencies
        - 'peak_indices': np.ndarray - Indices of peaks in frequency array
        - 'freqs': np.ndarray - Full frequency array (for reference)
        - 'magnitude': np.ndarray - Full magnitude spectrum (for reference)
        
    Examples:
    ---------
    >>> spectrum = compute_fft_spectrum(signal, sampling_rate=1000.0)
    >>> peaks = find_spectral_peaks(spectrum, prominence=0.1, num_peaks=5)
    >>> print(f"Top 5 peak frequencies: {peaks['peak_freqs']}")
    """
    freqs = spectrum_data['freqs']
    magnitude = spectrum_data['magnitude']
    
    # Find peaks using scipy's find_peaks
    peak_indices, properties = find_peaks(
        magnitude,
        height=height,
        prominence=prominence,
        distance=distance
    )
    
    # Sort by magnitude and take top peaks
    if len(peak_indices) > num_peaks:
        peak_magnitudes = magnitude[peak_indices]
        sorted_indices = np.argsort(peak_magnitudes)[::-1][:num_peaks]
        peak_indices = peak_indices[sorted_indices]
    
    # Sort peaks by frequency for consistent output
    peak_indices = np.sort(peak_indices)
    
    return {
        'peak_freqs': freqs[peak_indices],
        'peak_magnitudes': magnitude[peak_indices],
        'peak_indices': peak_indices,
        'freqs': freqs,
        'magnitude': magnitude
    }


def compute_spectrum_and_peaks(signal: np.ndarray,
                              sampling_rate: float,
                              nperseg: Optional[int] = None,
                              normalize: bool = True,
                              height: Optional[float] = None,
                              prominence: Optional[float] = None,
                              distance: Optional[int] = None,
                              num_peaks: int = 10) -> Dict[str, Any]:
    """
    Convenience function to compute spectrum and find peaks in one call.
    
    Parameters:
    -----------
    signal : np.ndarray
        Input signal to analyze
    sampling_rate : float
        Sampling rate of the signal in Hz
    nperseg : int, optional
        Length of each segment for Welch's method
    normalize : bool, default=True
        Whether to normalize the spectrum
    height, prominence, distance, num_peaks : 
        Peak detection parameters (see find_spectral_peaks)
        
    Returns:
    --------
    Dict[str, Any]
        Combined dictionary with spectrum and peak information
    """
    spectrum = compute_fft_spectrum(signal, sampling_rate, nperseg, normalize)
    peaks = find_spectral_peaks(spectrum, height, prominence, distance, num_peaks)
    
    # Merge dictionaries
    result = spectrum.copy()
    result.update({f'peak_{k}': v for k, v in peaks.items() if k not in ['freqs', 'magnitude']})
    
    return result