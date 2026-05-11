"""
Frequency-domain feature extraction for motor health monitoring data.

This module provides FFT-based spectral features commonly used
in motor health monitoring and fault diagnosis.
"""

import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class FrequencyDomainFeatures:
    """Extract frequency-domain features from signals."""

    @staticmethod
    def fft_features(signal: np.ndarray, sampling_rate: float) -> Dict[str, float]:
        """
        Extract FFT-based features with windowing to reduce spectral leakage.
        
        Args:
            signal: Input signal array
            sampling_rate: Sampling rate in Hz
        
        Returns:
            Tuple of (features_dict, fft_magnitude, fft_frequencies)
        """
        features = {}

        # Hann-windowed single-sided FFT (matches preprocessing preview)
        hann = np.hanning(len(signal))
        fft_freqs = np.fft.rfftfreq(len(signal), d=1.0 / sampling_rate)
        fft_magnitude = np.abs(np.fft.rfft(signal * hann)) / np.sum(hann)
        fft_magnitude = fft_magnitude.copy()
        fft_magnitude[1:-1] *= 2  # compensate for dropped negative freqs (except DC and Nyquist)

        # Spectral features
        features['spectral_centroid'] = np.sum(fft_freqs * fft_magnitude) / np.sum(fft_magnitude)
        variance = np.sum(((fft_freqs - features['spectral_centroid'])**2) * fft_magnitude) / np.sum(fft_magnitude)
        features['spectral_spread'] = np.sqrt(variance)
        features['spectral_rolloff'] = FrequencyDomainFeatures._spectral_rolloff(fft_magnitude, fft_freqs, 0.85)

        from scipy.stats import entropy
        se_scipy = entropy(fft_magnitude + 1e-12, base=2)
        features['spectral_entropy'] = float(se_scipy) / np.log2(len(fft_magnitude))

        # Harmonic ratio features: A2/A1 and A3/A1
        from ml_toolbox.signal_processing import find_dominant_frequency, find_harmonics
        dom_freq = find_dominant_frequency(fft_freqs, fft_magnitude)
        if dom_freq is not None:
            harmonics = find_harmonics(fft_freqs, fft_magnitude, f0=dom_freq, n_harmonics=3)
            amps = {h['harmonic_n']: h['amplitude'] for h in harmonics}
            a1 = amps.get(1, 0.0)
            features['a2_a1'] = float(amps.get(2, 0.0) / a1) if a1 > 0 else 0.0
            features['a3_a1'] = float(amps.get(3, 0.0) / a1) if a1 > 0 else 0.0
        else:
            features['a2_a1'] = 0.0
            features['a3_a1'] = 0.0

        return features

    @staticmethod
    def _spectral_rolloff(magnitude: np.ndarray, freqs: np.ndarray, threshold: float = 0.85) -> float:
        """Calculate spectral rolloff frequency."""
        total_energy = np.sum(magnitude**2)
        cumulative_energy = np.cumsum(magnitude**2)
        rolloff_idx = np.where(cumulative_energy >= threshold * total_energy)[0]
        return float(freqs[rolloff_idx[0]]) if len(rolloff_idx) > 0 else float(freqs[-1])
      