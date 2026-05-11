"""
Frequency-domain feature extraction for motor health monitoring data.

This module provides FFT-based spectral features commonly used
in motor health monitoring and fault diagnosis.
"""

import numpy as np
from typing import Dict, Tuple
from scipy.fft import fft, fftfreq
from scipy.signal import welch
import logging

logger = logging.getLogger(__name__)

class FrequencyDomainFeatures:
    """Extract frequency-domain features from signals."""
    
    @staticmethod
    def _apply_window(signal: np.ndarray, window_type: str = 'hann') -> np.ndarray:
        """Apply windowing function to reduce spectral leakage."""
        if window_type == 'hann':
            window = np.hanning(len(signal))
        elif window_type == 'hamming':
            window = np.hamming(len(signal))
        elif window_type == 'blackman':
            window = np.blackman(len(signal))
        elif window_type == 'none' or window_type is None:
            window = np.ones(len(signal))  # No windowing
        else:
            # Default to Hann window for unknown types
            window = np.hanning(len(signal))
        
        return signal * window
    
    @staticmethod
    def fft_features(signal: np.ndarray, sampling_rate: float, window_type: str = 'hann') -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """
        Extract FFT-based features with windowing to reduce spectral leakage.
        
        Args:
            signal: Input signal array
            sampling_rate: Sampling rate in Hz
            window_type: Window function type ('hann', 'hamming', 'blackman', 'none'). Default is 'hann'
        
        Returns:
            Tuple of (features_dict, fft_magnitude, fft_frequencies)
        """
        features = {}
        
        # Apply windowing to reduce spectral leakage
        #windowed_signal = FrequencyDomainFeatures._apply_window(signal, window_type)
        # Compute FFT
        #fft_vals = np.array(fft(windowed_signal))
        #freqs = fftfreq(len(windowed_signal), 1/sampling_rate)
        
        # Try to welch
        fft_freqs, fft_magnitude = welch(signal, fs=sampling_rate, nperseg=2048)

        # Only positive frequencies
        #n_positive = len(freqs) // 2
        #fft_magnitude = np.abs(fft_vals[:n_positive])
        #fft_freqs = freqs[:n_positive]
                 
        # Spectral features
        features['spectral_centroid'] = np.sum(fft_freqs * fft_magnitude) / np.sum(fft_magnitude)
        variance = np.sum(((fft_freqs - features['spectral_centroid'])**2) * fft_magnitude) / np.sum(fft_magnitude)
        features['spectral_spread'] = np.sqrt(variance)
        
        # Spectral energy
        features['spectral_rolloff'] = FrequencyDomainFeatures._spectral_rolloff(fft_magnitude, fft_freqs, 0.85)
        
        # Spectral entropy - measure of spectral complexity/randomness
        from scipy.stats import entropy
        se_scipy = entropy(fft_magnitude + 1e-12, base=2)
        features['spectral_entropy'] = float(se_scipy) / np.log2(len(fft_magnitude)) 
        
        return features, fft_magnitude, fft_freqs
    
    @staticmethod
    def _spectral_rolloff(magnitude: np.ndarray, freqs: np.ndarray, threshold: float = 0.85) -> float:
        """Calculate spectral rolloff frequency."""
        total_energy = np.sum(magnitude**2)
        cumulative_energy = np.cumsum(magnitude**2)
        rolloff_idx = np.where(cumulative_energy >= threshold * total_energy)[0]
        return float(freqs[rolloff_idx[0]]) if len(rolloff_idx) > 0 else float(freqs[-1])
      