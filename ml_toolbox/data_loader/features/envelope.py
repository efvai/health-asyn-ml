"""
Hilbert envelope feature extraction for motor health monitoring data.

This module provides Hilbert envelope analysis features commonly used
in motor health monitoring and fault diagnosis. It can reuse time-domain
and frequency-domain feature extraction for better scalability.
"""

import numpy as np
from typing import Dict, Optional
import logging
from ..envelope_analyzer import HilbertEnvelopeAnalyzer, EnvelopeConfig
from .base import ENV_CARRIER_FREQUENCY
from .time_domain import TimeDomainFeatures
from .frequency_domain import FrequencyDomainFeatures
from ...signal_processing.spectrum_analysis import compute_fft_spectrum, find_spectral_peaks

logger = logging.getLogger(__name__)
#logger.propagate = False

class HilbertEnvelopeFeatures:
    """Extract Hilbert envelope features from signals."""
    
    @staticmethod
    def _detect_carrier_frequency(signal: np.ndarray, 
                                  expected_freq: float = 1670.0,
                                  search_range: float = 200.0,
                                  sampling_rate: float = 10000.0) -> float:
        """
        Detect carrier frequency near expected value using spectral peak detection.
        
        Args:
            signal: Input signal
            expected_freq: Expected carrier frequency (Hz)
            search_range: Search range around expected frequency (Hz)
            sampling_rate: Sampling rate of the signal
            
        Returns:
            Detected carrier frequency
        """
        # Create a temporary config for initial spectrum analysis
        temp_conf = EnvelopeConfig(
            bandpass_low=expected_freq - search_range,
            bandpass_high=expected_freq + search_range,
            lowpass_cutoff=200.0,
            filter_order=4,
            decimation_factor=5,
            sampling_rate=sampling_rate
        )

        # Compute spectrum of original signal using centralized spectrum utilities
        spectrum = compute_fft_spectrum(signal, sampling_rate=temp_conf.sampling_rate, nperseg=2048, normalize=False)
        
        # Focus on frequency range around expected carrier
        freq_mask = (spectrum['freqs'] >= expected_freq - search_range) & \
                   (spectrum['freqs'] <= expected_freq + search_range)
        
        if not np.any(freq_mask):
            logger.warning(f"No frequencies found in range {expected_freq}±{search_range} Hz, using expected value")
            return expected_freq
        
        # Find peaks in the carrier frequency range
        carrier_spectrum = {
            'freqs': spectrum['freqs'][freq_mask],
            'magnitude': spectrum['magnitude'][freq_mask],
            'power': spectrum['power'][freq_mask]
        }
        
        # Use median-based peak detection
        median_mag = np.median(carrier_spectrum['magnitude'])
        peaks = find_spectral_peaks(
            carrier_spectrum,
            height=median_mag * 2,
            prominence=median_mag * 1.5,
            distance=10,
            num_peaks=5
        )
        
        if len(peaks['peak_indices']) > 0:
            # Return the strongest peak frequency
            strongest_idx = np.argmax(peaks['peak_magnitudes'])
            detected_freq = peaks['peak_freqs'][strongest_idx]
            #logger.info(f"Detected carrier frequency: {detected_freq:.1f} Hz (expected: {expected_freq:.1f} Hz)")
            return float(detected_freq)
        else:
            logger.warning(f"No carrier peak detected, using expected frequency {expected_freq} Hz")
            return expected_freq
    
    @staticmethod
    def hilbert_envelope_features(signal: np.ndarray, 
                                 bandpass_low: Optional[float] = None,
                                 bandpass_high: Optional[float] = None,
                                 expected_carrier: float = ENV_CARRIER_FREQUENCY,
                                 carrier_bandwidth: float = 50.0) -> Dict[str, float]:
        """
        Extract features from the Hilbert envelope of the signal.
        
        Args:
            signal: Input signal
            bandpass_low: Low frequency for bandpass filter (Hz). If None, auto-detect.
            bandpass_high: High frequency for bandpass filter (Hz). If None, auto-detect.
            expected_carrier: Expected carrier frequency for auto-detection (Hz)
            carrier_bandwidth: Bandwidth around carrier (±Hz)
            
        Returns:
            Dictionary of Hilbert envelope features
        """
        features = {}
        
        # Auto-detect carrier frequency if bandpass not specified
        if bandpass_low is None or bandpass_high is None:
            carrier_freq = HilbertEnvelopeFeatures._detect_carrier_frequency(
                signal, 
                expected_freq=expected_carrier,
                search_range=200.0,
                sampling_rate=10000.0
            )
            bandpass_low = carrier_freq - carrier_bandwidth
            bandpass_high = carrier_freq + carrier_bandwidth
        
        env_conf = EnvelopeConfig(
            bandpass_low=bandpass_low,
            bandpass_high=bandpass_high,
            lowpass_cutoff=200.0,
            filter_order=4,
            decimation_factor=5,
            sampling_rate=10000 # current fs
        )
        analyzer = HilbertEnvelopeAnalyzer(env_conf)
        envelope = analyzer.extract_envelope(signal)
    
        envelope_time_features = TimeDomainFeatures.basic_statistics(envelope)
        for key, value in envelope_time_features.items():
            features[f'env_{key}'] = value
        
        # Compute spectrum of the decimated envelope (use envelope sampling rate)
        env_spectrum = compute_fft_spectrum(envelope, sampling_rate=analyzer.config.envelope_fs, nperseg=512, normalize=False)
        
        # Extract frequency-domain features from envelope spectrum
        spectral_features = FrequencyDomainFeatures.spectral_analysis_features(env_spectrum)
        for key, value in spectral_features.items():
            features[f'env_{key}'] = value

        # Find peaks 
        peak_cutoff = 200  # Hz
        cutoff_idx_h = np.where(env_spectrum['freqs'] <= peak_cutoff)[0][-1] if len(np.where(env_spectrum['freqs'] <= peak_cutoff)[0]) > 0 else len(env_spectrum['freqs'])//2
        median = np.median(env_spectrum['magnitude'][:cutoff_idx_h])
        peaks = find_spectral_peaks(env_spectrum, height=median*2, prominence=median, distance=2, num_peaks=10)
        
        # Extract peak analysis features
        peak_features = FrequencyDomainFeatures.peak_analysis_features(env_spectrum, peaks, peak_cutoff)
        for key, value in peak_features.items():
            features[f'env_{key}'] = value
        
        # Add THD-like harmonic analysis features using found peaks
        peak_count = len(peaks["peak_indices"])
        if peak_count > 0:
            peak_freqs = peaks["peak_freqs"]
            peak_mags = peaks["peak_magnitudes"]
            
            # Use dominant peak as fundamental frequency
            dom_idx = np.argmax(peak_mags)
            f0 = peak_freqs[dom_idx]
            
            # Extract harmonic analysis features
            harmonic_features = FrequencyDomainFeatures.harmonic_analysis_features(
                env_spectrum, peaks, f0, max_harmonics=5
            )
            for key, value in harmonic_features.items():
                features[f'env_{key}'] = value
        else:
            # Add zero values for harmonic features when no peaks found
            harmonic_features = FrequencyDomainFeatures.harmonic_analysis_features(
                env_spectrum, peaks, 0.0, max_harmonics=5
            )
            for key, value in harmonic_features.items():
                features[f'env_{key}'] = value

        return features
    
    @staticmethod
    def hilbert_envelope_cross_features(signal_a: np.ndarray, signal_b: np.ndarray, 
                                        ch1_name: str = "ch1", ch2_name: str = "ch2",
                                        bandpass_low: Optional[float] = None,
                                        bandpass_high: Optional[float] = None,
                                        expected_carrier: float = ENV_CARRIER_FREQUENCY,
                                        carrier_bandwidth: float = 50.0) -> Dict[str, float]:
        """
        Extract cross-channel Hilbert envelope features.
        
        Args:
            signal_a: First channel signal
            signal_b: Second channel signal
            ch1_name: Name of first channel
            ch2_name: Name of second channel
            bandpass_low: Low frequency for bandpass filter (Hz). If None, auto-detect.
            bandpass_high: High frequency for bandpass filter (Hz). If None, auto-detect.
            expected_carrier: Expected carrier frequency for auto-detection (Hz)
            carrier_bandwidth: Bandwidth around carrier (±Hz)
            
        Returns:
            Dictionary of cross-channel envelope features
        """
        features = {}

        # Auto-detect carrier frequency if bandpass not specified
        # Use signal_a for detection (could also average both signals)
        if bandpass_low is None or bandpass_high is None:
            carrier_freq = HilbertEnvelopeFeatures._detect_carrier_frequency(
                signal_a, 
                expected_freq=expected_carrier,
                search_range=200.0,
                sampling_rate=10000.0
            )
            bandpass_low = carrier_freq - carrier_bandwidth
            bandpass_high = carrier_freq + carrier_bandwidth
        
        env_conf = EnvelopeConfig(
            bandpass_low=bandpass_low,
            bandpass_high=bandpass_high,
            lowpass_cutoff=200.0,
            filter_order=4,
            decimation_factor=5,
            sampling_rate=10000 # current fs
        )
        analyzer = HilbertEnvelopeAnalyzer(env_conf)

        envelope1 = analyzer.extract_envelope(signal_a)
        envelope2 = analyzer.extract_envelope(signal_b)
        
        # Use TimeDomainFeatures for cross-correlation analysis
        cross_time_features = TimeDomainFeatures.cross_correlation_features(
            envelope1, envelope2, f"{ch1_name}_env", f"{ch2_name}_env"
        )
        # Rename the features to match expected naming convention  
        for key, value in cross_time_features.items():
            new_key = key.replace(f"{ch1_name}_env_{ch2_name}_env", f"{ch1_name}_{ch2_name}_env")
            features[new_key] = value

        # Compute spectra of decimated envelopes for both channels
        env_s1 = compute_fft_spectrum(envelope1, sampling_rate=analyzer.config.envelope_fs, nperseg=512, normalize=True)
        env_s2 = compute_fft_spectrum(envelope2, sampling_rate=analyzer.config.envelope_fs, nperseg=512, normalize=True)
        
        # Use FrequencyDomainFeatures for cross-spectral analysis
        cross_spectral_features = FrequencyDomainFeatures.cross_spectral_features(
            env_s1, env_s2, f"{ch1_name}_env", f"{ch2_name}_env"
        )
        # Rename the features to match expected naming convention
        for key, value in cross_spectral_features.items():
            new_key = key.replace(f"{ch1_name}_env_{ch2_name}_env", f"{ch1_name}_{ch2_name}_env")
            features[new_key] = value
        
        # Find peaks for magnitude comparison
        peak_cutoff = 200  # Hz
        cutoff_idx1 = np.where(env_s1['freqs'] <= peak_cutoff)[0][-1] if len(np.where(env_s1['freqs'] <= peak_cutoff)[0]) > 0 else len(env_s1['freqs'])//2
        median1 = np.median(env_s1['magnitude'][:cutoff_idx1])
        peaks1 = find_spectral_peaks(env_s1, height=median1*2, prominence=median1, distance=2, num_peaks=10)
        
        cutoff_idx2 = np.where(env_s2['freqs'] <= peak_cutoff)[0][-1] if len(np.where(env_s2['freqs'] <= peak_cutoff)[0]) > 0 else len(env_s2['freqs'])//2
        median2 = np.median(env_s2['magnitude'][:cutoff_idx2])
        peaks2 = find_spectral_peaks(env_s2, height=median2*2, prominence=median2, distance=2, num_peaks=10)
        
        # Use FrequencyDomainFeatures for peak magnitude comparison
        peak_comparison_features = FrequencyDomainFeatures.peak_magnitude_comparison(
            peaks1, peaks2, f"{ch1_name}_env", f"{ch2_name}_env"
        )
        # Rename the features to match expected naming convention
        for key, value in peak_comparison_features.items():
            new_key = key.replace(f"{ch1_name}_env_{ch2_name}_env", f"{ch1_name}_{ch2_name}_env")
            features[new_key] = value

        return features