"""
Feature extraction module for motor health monitoring data.

This module provides comprehensive feature extraction capabilities for time series data,
including time-domain, frequency-domain, and time-frequency features commonly used
in motor health monitoring and fault diagnosis.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass
import logging
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert
import warnings

# Try to import PCA reduction module
try:
    from .pca_reduction import PCAFeatureReducer, PCAConfig, create_pca_config_for_time_domain
    PCA_AVAILABLE = True
except ImportError:
    PCA_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("PCA reduction module not available. Install scikit-learn for PCA functionality.")

logger = logging.getLogger(__name__)

# Sensor-specific sampling rates
CURRENT_SAMPLING_RATE = 10000.0   # LTR11 - Current sensors
VIBRATION_SAMPLING_RATE = 26041.0 # LTR22 - Vibration sensors


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    sampling_rate: float = CURRENT_SAMPLING_RATE  # Default for current sensors (LTR11)
    
    # Time domain features
    time_domain: bool = True
    statistical_moments: bool = True
    shape_factors: bool = True
    
    # Frequency domain features
    frequency_domain: bool = True
    fft_features: bool = True
    spectral_features: bool = True
    
    # PCA feature reduction
    apply_pca: bool = False  # Apply PCA to reduce dimensionality of time-domain features
    pca_variance_threshold: float = 0.95  # Variance threshold for PCA component selection
    pca_n_components: Optional[int] = None  # Fixed number of components (None for auto)
    
    # Frequency analysis parameters
    max_frequency: Optional[float] = None
    frequency_bands: Optional[List[tuple]] = None
    
    # Windowing parameters for spectral leakage reduction
    window_type: str = 'hann'  # Options: 'hann', 'hamming', 'blackman', 'bartlett', 'kaiser', 'none'
    
    def __post_init__(self):
        """Set default parameters."""
        if self.max_frequency is None:
            self.max_frequency = self.sampling_rate / 2  # Nyquist frequency
        
        if self.frequency_bands is None:
            # Default frequency bands for motor analysis
            self.frequency_bands = [
                (0, 100),      
                (100, 500),    
                (500, 2000),   
                (2000, 3500)   
            ]


class TimeDomainFeatures:
    """Extract time-domain features from signals."""
    
    @staticmethod
    def basic_statistics(signal: np.ndarray) -> Dict[str, float]:
        """Extract basic statistical features."""
        features = {}
        
        # Basic statistics
        features['rms'] = np.sqrt(np.mean(signal**2))
        features['peak_to_peak'] = np.ptp(signal)
        
        # Percentiles
        features['iqr'] = np.percentile(signal, 75) - np.percentile(signal, 25)
        
        return features
    
    @staticmethod
    def statistical_moments(signal: np.ndarray) -> Dict[str, float]:
        """Extract higher-order statistical moments."""
        features = {}
        
        
        features['rms'] = np.sqrt(np.mean(signal**2))
        features['skewness'] = stats.skew(signal)
        features['kurtosis'] = stats.kurtosis(signal)
        
        return features
    
    @staticmethod
    def shape_factors(signal: np.ndarray) -> Dict[str, float]:
        """Extract shape factor features."""
        features = {}
        
        rms = np.sqrt(np.mean(signal**2))
        peak = np.max(np.abs(signal))
        mean_abs = np.mean(np.abs(signal))
        
        # Avoid division by zero
        eps = 1e-12
        
        features['crest_factor'] = peak / (rms + eps)
        features['form_factor'] = rms / (mean_abs + eps)
        
        return features

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
    def fft_features(signal: np.ndarray, sampling_rate: float, window_type: str = 'hann') -> tuple:
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
        from scipy.signal import welch
        fft_freqs, fft_magnitude = welch(signal, fs=sampling_rate, nperseg=2048)

        # Only positive frequencies
        #n_positive = len(freqs) // 2
        #fft_magnitude = np.abs(fft_vals[:n_positive])
        #fft_freqs = freqs[:n_positive]
        
        # Normalize FFT magnitude
        total_energy = np.sum(fft_magnitude**2)
        if total_energy > 1e-12:
            # Normalize by total spectral energy
            fft_magnitude_normalized = fft_magnitude / np.sqrt(total_energy)
        else:
            fft_magnitude_normalized = fft_magnitude
                
        # Spectral features
        features['spectral_centroid'] = np.sum(fft_freqs * fft_magnitude_normalized) / (np.sum(fft_magnitude_normalized) + 1e-12)
        features['spectral_spread'] = np.sqrt(np.sum((fft_freqs - features['spectral_centroid'])**2 * fft_magnitude_normalized) / (np.sum(fft_magnitude_normalized) + 1e-12))
        
        # Spectral energy
        features['spectral_rolloff'] = FrequencyDomainFeatures._spectral_rolloff(fft_magnitude, fft_freqs, 0.85)
        features['spectral_energy'] = np.sum(fft_magnitude**2)
        
        # Spectral entropy - measure of spectral complexity/randomness
        power_spectrum = fft_magnitude**2
        power_spectrum_norm = power_spectrum / (np.sum(power_spectrum) + 1e-12)
        power_spectrum_norm = power_spectrum_norm[power_spectrum_norm > 1e-12]  # Remove near-zeros
        if len(power_spectrum_norm) > 0:
            features['spectral_entropy'] = float(-np.sum(power_spectrum_norm * np.log2(power_spectrum_norm + 1e-12)))
        else:
            features['spectral_entropy'] = 0.0
        
        # Spectral flatness (Wiener entropy) - measure of how noise-like vs tone-like the spectrum is
        # Ratio of geometric mean to arithmetic mean of power spectrum
        power_spectrum_positive = power_spectrum[power_spectrum > 1e-12]
        if len(power_spectrum_positive) > 0:
            geometric_mean = np.exp(np.mean(np.log(power_spectrum_positive + 1e-12)))
            arithmetic_mean = np.mean(power_spectrum_positive)
            features['spectral_flatness'] = float(geometric_mean / (arithmetic_mean + 1e-12))
        else:
            features['spectral_flatness'] = 0.0
        
        return features, fft_magnitude, fft_freqs
    
    @staticmethod
    def _spectral_rolloff(magnitude: np.ndarray, freqs: np.ndarray, threshold: float = 0.85) -> float:
        """Calculate spectral rolloff frequency."""
        total_energy = np.sum(magnitude**2)
        cumulative_energy = np.cumsum(magnitude**2)
        rolloff_idx = np.where(cumulative_energy >= threshold * total_energy)[0]
        return float(freqs[rolloff_idx[0]]) if len(rolloff_idx) > 0 else float(freqs[-1])
    
    @staticmethod
    def spectral_bands_features(magnitude: np.ndarray, freqs: np.ndarray, bands: List[tuple]) -> Dict[str, float]:
        """Extract features from specific frequency bands."""
        features = {}
    
        for i, (low_freq, high_freq) in enumerate(bands):
            band_name = f"band_{i+1}_{low_freq}_{high_freq}hz"
            
            # Find indices for frequency band
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)
            band_magnitude = magnitude[band_mask]
            band_freqs = freqs[band_mask]
            
            if len(band_magnitude) > 0:
                # Energy and peak features
                features[f'{band_name}_energy'] = np.sum(band_magnitude**2)
                features[f'{band_name}_power_ratio'] = features[f'{band_name}_energy'] / (np.sum(magnitude**2) + 1e-12)
                
                # Normalize band magnitude for centroid and spread calculations
                total_band_energy = np.sum(band_magnitude**2)
                if total_band_energy > 1e-12:
                    band_magnitude_normalized = band_magnitude / np.sqrt(total_band_energy)
                else:
                    band_magnitude_normalized = band_magnitude
                
                # Spectral centroid for this band
                features[f'{band_name}_centroid'] = np.sum(band_freqs * band_magnitude_normalized) / (np.sum(band_magnitude_normalized) + 1e-12)
                
                # Spectral spread for this band
                centroid = features[f'{band_name}_centroid']
                features[f'{band_name}_spread'] = np.sqrt(np.sum((band_freqs - centroid)**2 * band_magnitude_normalized) / (np.sum(band_magnitude_normalized) + 1e-12))
                
                # Spectral entropy for this band
                power_spectrum = band_magnitude**2
                power_spectrum_norm = power_spectrum / (np.sum(power_spectrum) + 1e-12)
                power_spectrum_norm = power_spectrum_norm[power_spectrum_norm > 1e-12]  # Remove near-zeros
                if len(power_spectrum_norm) > 0:
                    features[f'{band_name}_entropy'] = float(-np.sum(power_spectrum_norm * np.log2(power_spectrum_norm + 1e-12)))
                else:
                    features[f'{band_name}_entropy'] = 0.0
                
                # Spectral flatness for this band
                power_spectrum_positive = power_spectrum[power_spectrum > 1e-12]
                if len(power_spectrum_positive) > 0:
                    geometric_mean = np.exp(np.mean(np.log(power_spectrum_positive + 1e-12)))
                    arithmetic_mean = np.mean(power_spectrum_positive)
                    features[f'{band_name}_flatness'] = float(geometric_mean / (arithmetic_mean + 1e-12))
                else:
                    features[f'{band_name}_flatness'] = 0.0
                
                # Band-specific peak detection features
                peak_features = FrequencyDomainFeatures.spectral_peaks_features(band_magnitude, band_freqs, band_name)
                features.update(peak_features)
                
            else:
                features[f'{band_name}_energy'] = 0.0
                features[f'{band_name}_peak'] = 0.0
                features[f'{band_name}_power_ratio'] = 0.0
                features[f'{band_name}_centroid'] = 0.0
                features[f'{band_name}_spread'] = 0.0
                features[f'{band_name}_entropy'] = 0.0
                features[f'{band_name}_flatness'] = 0.0
                # Zero peak features for empty bands
                features[f'{band_name}_peak_count'] = 0.0
                features[f'{band_name}_peak_prominence_mean'] = 0.0
                features[f'{band_name}_peak_prominence_std'] = 0.0
                features[f'{band_name}_peak_power_mean'] = 0.0
                features[f'{band_name}_peak_power_std'] = 0.0
                features[f'{band_name}_dominant_peak_freq'] = 0.0
                features[f'{band_name}_dominant_peak_power'] = 0.0
                features[f'{band_name}_peak_density'] = 0.0
                features[f'{band_name}_dominant_relative_peak_power'] = 0.0
                features[f'{band_name}_dominant_peak_to_band_energy_ratio'] = 0.0
                features[f'{band_name}_peak_spacing_mean'] = 0.0
                features[f'{band_name}_peak_spacing_std'] = 0.0
                features[f'{band_name}_peak_freq_variance'] = 0.0
                features[f'{band_name}_peak_freq_std'] = 0.0
                features[f'{band_name}_peak_freq_range'] = 0.0
                features[f'{band_name}_peak_freq_cv'] = 0.0
        
        return features

    @staticmethod
    def spectral_peaks_features(magnitude: np.ndarray, freqs: np.ndarray, band_name: str) -> Dict[str, float]:
        """
        Extract peak-related features using band-specific peak detection.
        
        Args:
            magnitude: Magnitude spectrum for the band
            freqs: Frequency array for the band
            band_name: Name prefix for features
            
        Returns:
            Dictionary of peak-related features
        """
        features = {}
        
        # Maybe Better to use normalized magnitude?
        # Need to investigate later
  
        from scipy.signal import find_peaks
            
        # Find peaks using band-specific thresholds
        magniture_norm = magnitude / (np.max(magnitude) + 1e-12)
        noise_floor = np.median(magniture_norm)
        peak_indices, peak_properties = find_peaks(
            magniture_norm,
            height=noise_floor * 3,      # threshold above band baseline
            prominence=noise_floor * 2,  # prominence relative to band baseline
            distance=5                   # minimum distance between peaks
        )
        
        # Basic peak count
        peak_count = len(peak_indices)
        features[f'{band_name}_peak_count'] = float(peak_count)
        
        if peak_count > 0:
            # Peak frequencies and powers
            peak_freqs = freqs[peak_indices]
            peak_powers = magnitude[peak_indices]
            peak_prominences = peak_properties.get('prominences', np.zeros(peak_count))
            
            # Calculate total band power and energy for normalization
            total_band_power = np.sum(magnitude**2)
            total_band_energy = np.sum(magnitude**2)  # Same as power for magnitude spectrum
            
            # Peak prominence statistics
            features[f'{band_name}_peak_prominence_mean'] = float(np.mean(peak_prominences))
            features[f'{band_name}_peak_prominence_std'] = float(np.std(peak_prominences))
            
            # Peak power statistics
            features[f'{band_name}_peak_power_mean'] = float(np.mean(peak_powers))
            features[f'{band_name}_peak_power_std'] = float(np.std(peak_powers))
            
            # Dominant peak (highest power)
            dominant_idx = np.argmax(peak_powers)
            dominant_peak_power = peak_powers[dominant_idx]
            features[f'{band_name}_dominant_peak_freq'] = float(peak_freqs[dominant_idx])
            features[f'{band_name}_dominant_peak_power'] = float(dominant_peak_power)
            
            # 1. Dominant Relative peak power: peak_power / total_band_power
            features[f'{band_name}_dominant_relative_peak_power'] = float(dominant_peak_power**2 / (total_band_power + 1e-12))
            
            # 2. Dominant Peak-to-band energy ratio: peak_power / (sum of all PSD in band)
            features[f'{band_name}_dominant_peak_to_band_energy_ratio'] = float(dominant_peak_power**2 / (total_band_energy + 1e-12))
                    
            # 4. Peak spacing / average distance: average difference in frequency between consecutive peaks
            if peak_count > 1:
                sorted_peak_freqs = np.sort(peak_freqs)
                peak_spacings = np.diff(sorted_peak_freqs)
                features[f'{band_name}_peak_spacing_mean'] = float(np.mean(peak_spacings))
                features[f'{band_name}_peak_spacing_std'] = float(np.std(peak_spacings))
            else:
                features[f'{band_name}_peak_spacing_mean'] = 0.0
                features[f'{band_name}_peak_spacing_std'] = 0.0
            
            # 5. Peak dispersion / variance: variability in peak frequencies
            if peak_count > 1:
                features[f'{band_name}_peak_freq_variance'] = float(np.var(peak_freqs))
                features[f'{band_name}_peak_freq_std'] = float(np.std(peak_freqs))
                features[f'{band_name}_peak_freq_range'] = float(np.max(peak_freqs) - np.min(peak_freqs))
                
                # Coefficient of variation for peak frequencies (normalized dispersion)
                mean_peak_freq = np.mean(peak_freqs)
                if mean_peak_freq > 1e-12:
                    features[f'{band_name}_peak_freq_cv'] = float(np.std(peak_freqs) / mean_peak_freq)
                else:
                    features[f'{band_name}_peak_freq_cv'] = 0.0
            else:
                features[f'{band_name}_peak_freq_variance'] = 0.0
                features[f'{band_name}_peak_freq_std'] = 0.0
                features[f'{band_name}_peak_freq_range'] = 0.0
                features[f'{band_name}_peak_freq_cv'] = 0.0
            
            # Peak density (peaks per Hz)
            freq_range = freqs[-1] - freqs[0] if len(freqs) > 1 else 1.0
            features[f'{band_name}_peak_density'] = float(peak_count / max(freq_range, 1.0))
            
        else:
            # No peaks detected
            features[f'{band_name}_peak_prominence_mean'] = 0.0
            features[f'{band_name}_peak_prominence_std'] = 0.0
            features[f'{band_name}_peak_power_mean'] = 0.0
            features[f'{band_name}_peak_power_std'] = 0.0
            features[f'{band_name}_dominant_peak_freq'] = 0.0
            features[f'{band_name}_dominant_peak_power'] = 0.0
            features[f'{band_name}_peak_density'] = 0.0
            
            # New requested features - zero values
            features[f'{band_name}_dominant_relative_peak_power'] = 0.0
            features[f'{band_name}_dominant_peak_to_band_energy_ratio'] = 0.0
            features[f'{band_name}_peak_spacing_mean'] = 0.0
            features[f'{band_name}_peak_spacing_std'] = 0.0
            features[f'{band_name}_peak_freq_variance'] = 0.0
            features[f'{band_name}_peak_freq_std'] = 0.0
            features[f'{band_name}_peak_freq_range'] = 0.0
            features[f'{band_name}_peak_freq_cv'] = 0.0
        
        return features

class FeatureExtractor:
    """Main feature extraction class."""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.time_domain = TimeDomainFeatures()
        self.frequency_domain = FrequencyDomainFeatures()
    
    def extract_features(self, signal: np.ndarray, channel_name: str = "ch") -> Dict[str, float]:
        """
        Extract comprehensive features from a single-channel signal.
        
        Args:
            signal: 1D signal array
            channel_name: Name prefix for features
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Time domain features
        if self.config.time_domain:
            time_features = self.time_domain.basic_statistics(signal)
            features.update({f"{channel_name}_{k}": v for k, v in time_features.items()})
                   
        if self.config.statistical_moments:
            moment_features = self.time_domain.statistical_moments(signal)
            features.update({f"{channel_name}_{k}": v for k, v in moment_features.items()})
        
        if self.config.shape_factors:
            shape_features = self.time_domain.shape_factors(signal)
            features.update({f"{channel_name}_{k}": v for k, v in shape_features.items()})
        
        # Frequency domain features
        if self.config.frequency_domain:
            fft_result = self.frequency_domain.fft_features(
                signal, self.config.sampling_rate, self.config.window_type
            )
            fft_features, magnitude, freqs = fft_result
            features.update({f"{channel_name}_{k}": v for k, v in fft_features.items()})
            
            if self.config.spectral_features and self.config.frequency_bands:
                band_features = self.frequency_domain.spectral_bands_features(
                    magnitude, freqs, self.config.frequency_bands
                )
                features.update({f"{channel_name}_{k}": v for k, v in band_features.items()})
        
        return features
    
    def extract_features_multichannel(self, signal: np.ndarray, 
                                    channel_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Extract features from multi-channel signal.
        
        Args:
            signal: 2D array (samples, channels)
            channel_names: Names for each channel
            
        Returns:
            Dictionary of extracted features
        """
        if len(signal.shape) != 2:
            raise ValueError("Signal must be 2D (samples, channels)")
        
        n_samples, n_channels = signal.shape
        
        if channel_names is None:
            channel_names = [f"ch{i}" for i in range(n_channels)]
        
        all_features = {}
        
        # Extract features for each channel

        # Get only first channel of current (TEMP)
        only_one_phase = ["current_phase_a"]

        for ch_idx, ch_name in enumerate(only_one_phase):
            ch_signal = signal[:, ch_idx]
            ch_features = self.extract_features(ch_signal, ch_name)
            all_features.update(ch_features)
        
        # Cross-channel features
        if n_channels > 1:
            cross_features = self._extract_cross_channel_features(signal, channel_names)
            all_features.update(cross_features)
        
        return all_features
    
    def _extract_cross_channel_features(self, signal: np.ndarray, 
                                      channel_names: List[str]) -> Dict[str, float]:
        """Extract cross-channel correlation and coherence features."""
        features = {}
        n_channels = signal.shape[1]
        
        # Cross-correlation features
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                ch1_name = channel_names[i]
                ch2_name = channel_names[j]
                
                # Correlation coefficient
                #corr = np.corrcoef(signal[:, i], signal[:, j])[0, 1]
                #features[f"{ch1_name}_{ch2_name}_correlation"] = corr if not np.isnan(corr) else 0.0
                
                # Advanced cross-channel features using PSD analysis
                cross_psd_features = self._extract_cross_psd_features(
                    signal[:, i], signal[:, j], ch1_name, ch2_name
                )
                features.update(cross_psd_features)
        
        return features
    
    def _extract_cross_psd_features(self, signal_a: np.ndarray, signal_b: np.ndarray,
                                   ch1_name: str, ch2_name: str) -> Dict[str, float]:
        """Extract cross-channel features based on PSD analysis and coherence."""
        from scipy.signal import welch, find_peaks, coherence
        
        features = {}
        fs = self.config.sampling_rate
        
        try:
            # Compute PSDs using same parameters as fft_features method
            fA, psdA = welch(signal_a, fs=fs, nperseg=2048)
            fB, psdB = welch(signal_b, fs=fs, nperseg=2048)
            
            # Find peaks using similar approach as spectral_peaks_features
            # Normalize PSDs for peak detection
            psdA_norm = psdA / (np.max(psdA) + 1e-12)
            psdB_norm = psdB / (np.max(psdB) + 1e-12)
            
            noise_floor_A = np.median(psdA_norm)
            noise_floor_B = np.median(psdB_norm)
            
            peaksA, _ = find_peaks(
                psdA_norm,
                height=noise_floor_A * 3,
                prominence=noise_floor_A * 2,
                distance=5
            )
            peaksB, _ = find_peaks(
                psdB_norm,
                height=noise_floor_B * 3,
                prominence=noise_floor_B * 2,
                distance=5
            )
            
            # Get top peaks sorted by power
            if len(peaksA) > 0 and len(peaksB) > 0:
                topA = sorted(zip(fA[peaksA], psdA[peaksA]), key=lambda x: x[1], reverse=True)
                topB = sorted(zip(fB[peaksB], psdB[peaksB]), key=lambda x: x[1], reverse=True)
                
                fA1, pA1 = topA[0]
                fB1, pB1 = topB[0]
                
                # Cross-phase features
                diff_power = abs(pA1 - pB1)
                ratio_power = pA1 / (pB1 + 1e-12)
                diff_freq = abs(fA1 - fB1)
                ratio_freq = fA1 / (fB1 + 1e-12)
                
                features[f"{ch1_name}_{ch2_name}_cross_diff_power"] = float(diff_power)
                features[f"{ch1_name}_{ch2_name}_cross_ratio_power"] = float(ratio_power)
                features[f"{ch1_name}_{ch2_name}_cross_diff_freq"] = float(diff_freq)
                features[f"{ch1_name}_{ch2_name}_cross_ratio_freq"] = float(ratio_freq)
            else:
                # No peaks found in one or both channels
                features[f"{ch1_name}_{ch2_name}_cross_diff_power"] = 0.0
                features[f"{ch1_name}_{ch2_name}_cross_ratio_power"] = 1.0
                features[f"{ch1_name}_{ch2_name}_cross_diff_freq"] = 0.0
                features[f"{ch1_name}_{ch2_name}_cross_ratio_freq"] = 1.0
            
            # Coherence analysis
            freqs_coh, coh = coherence(signal_a, signal_b, fs=fs, nperseg=2048)
            mean_coh = np.mean(coh)
            
            # Peak coherence at dominant frequency
            if len(peaksA) > 0:
                # Find coherence at dominant frequency of channel A
                dominant_freq_idx = np.argmax(psdA)
                # Find closest frequency in coherence array
                freq_diff = np.abs(freqs_coh - fA[dominant_freq_idx])
                closest_idx = np.argmin(freq_diff)
                peak_coh = coh[closest_idx]
            else:
                peak_coh = mean_coh  # Fallback to mean coherence
            
            features[f"{ch1_name}_{ch2_name}_cross_mean_coherence"] = float(mean_coh)
            features[f"{ch1_name}_{ch2_name}_cross_peak_coherence"] = float(peak_coh)
            
        except Exception as e:
            logger.warning(f"Failed to extract cross-PSD features for {ch1_name}_{ch2_name}: {e}")
            # Fallback values
            features[f"{ch1_name}_{ch2_name}_cross_diff_power"] = 0.0
            features[f"{ch1_name}_{ch2_name}_cross_ratio_power"] = 1.0
            features[f"{ch1_name}_{ch2_name}_cross_diff_freq"] = 0.0
            features[f"{ch1_name}_{ch2_name}_cross_ratio_freq"] = 1.0
            features[f"{ch1_name}_{ch2_name}_cross_mean_coherence"] = 0.0
            features[f"{ch1_name}_{ch2_name}_cross_peak_coherence"] = 0.0
        
        return features
    
    def extract_features_batch(self, windows: np.ndarray, 
                             channel_names: Optional[List[str]] = None) -> tuple:
        """
        Extract features from a batch of windows.
        
        Args:
            windows: 3D array (n_windows, window_size, n_channels)
            channel_names: Names for each channel
            
        Returns:
            2D feature array (n_windows, n_features)
        """
        if len(windows.shape) != 3:
            raise ValueError("Windows must be 3D (n_windows, window_size, n_channels)")
        
        n_windows, window_size, n_channels = windows.shape
        
        if channel_names is None:
            channel_names = [f"ch{i}" for i in range(n_channels)]
        
        # Extract features from first window to get feature names
        sample_features = self.extract_features_multichannel(windows[0], channel_names)
        feature_names = list(sample_features.keys())
        n_features = len(feature_names)
        
        # Pre-allocate feature matrix
        feature_matrix = np.zeros((n_windows, n_features))
        
        # Extract features for all windows
        for i in range(n_windows):
            window_features = self.extract_features_multichannel(windows[i], channel_names)
            
            # Check for missing features and handle them
            feature_values = []
            missing_features = []
            for name in feature_names:
                if name in window_features:
                    feature_values.append(window_features[name])
                else:
                    feature_values.append(0.0)  # Default value for missing features
                    missing_features.append(name)
            
            if missing_features:
                logger.warning(f"Window {i}: Missing features {missing_features[:5]}{'...' if len(missing_features) > 5 else ''} (total: {len(missing_features)})")
            
            feature_matrix[i, :] = feature_values
            
            if i % 100 == 0:  # Progress logging
                logger.info(f"Processed {i}/{n_windows} windows")
        
        return feature_matrix, feature_names


def extract_categorical_features(metadata_list: List[Dict]) -> Tuple[np.ndarray, List[str]]:
    """
    Extract categorical features from metadata.
    
    Args:
        metadata_list: List of metadata dictionaries
        
    Returns:
        Tuple of (categorical_features, feature_names)
    """
    if not metadata_list:
        return np.array([]).reshape(0, 0), []
    
    n_windows = len(metadata_list)
    categorical_features = []
    feature_names = []
    
    # Extract numerical frequency value (for regression-style features)
    freq_values = []
    for meta in metadata_list:
        freq_str = meta.get('frequency', '0hz')
        # Extract number from frequency string (e.g., "20hz" -> 20)
        import re
        match = re.search(r'(\d+)', freq_str)
        freq_val = float(match.group(1)) if match else 0.0
        freq_values.append(freq_val)
    
    categorical_features.append(freq_values)
    feature_names.append('frequency_hz')
    
    # Extract load condition as single binary feature (1 = under_load, 0 = no_load)
    load_values = []
    for meta in metadata_list:
        load = meta.get('load', 'unknown')
        # Convert to binary: 1 for under_load, 0 for no_load or unknown
        load_val = 1 if load == 'under_load' else 0
        load_values.append(load_val)
    
    categorical_features.append(load_values)
    feature_names.append('load_under_load')
    
    # Extract sensor type features if available
    sensor_types = set()
    for meta in metadata_list:
        sensor = meta.get('sensor_type', 'unknown')
        sensor_types.add(sensor)
    
    if len(sensor_types) > 1:  # Only add if there are multiple sensor types
        sensor_types = sorted(list(sensor_types))
        for sensor in sensor_types:
            sensor_feature = [1 if meta.get('sensor_type') == sensor else 0 for meta in metadata_list]
            categorical_features.append(sensor_feature)
            feature_names.append(f'sensor_{sensor}')
    
    # Convert to numpy array
    if categorical_features:
        categorical_matrix = np.array(categorical_features).T
    else:
        categorical_matrix = np.array([]).reshape(n_windows, 0)
    
    return categorical_matrix, feature_names


def extract_features_for_ml(windows: np.ndarray, 
                           sampling_rate: Optional[float] = None,
                           sensor_type: str = "current",
                           feature_config: Optional[FeatureConfig] = None,
                           metadata_list: Optional[List[Dict]] = None) -> tuple:
    """
    Convenience function to extract features ready for ML.
    
    Args:
        windows: 3D array (n_windows, window_size, n_channels)
        sampling_rate: Sampling rate in Hz (auto-detected if None)
        sensor_type: Type of sensor ("current" or "vibration")
        feature_config: Custom feature configuration
        metadata_list: Optional list of metadata dicts for categorical features
        
    Returns:
        Tuple of (feature_matrix, feature_names, pca_reducer_if_used)
    """
    # Auto-detect sampling rate based on sensor type if not provided
    if sampling_rate is None:
        if sensor_type == "current":
            sampling_rate = CURRENT_SAMPLING_RATE  # LTR11
        elif sensor_type == "vibration":
            sampling_rate = VIBRATION_SAMPLING_RATE  # LTR22
        else:
            sampling_rate = CURRENT_SAMPLING_RATE  # Default fallback
            logger.warning(f"Unknown sensor type '{sensor_type}', using default sampling rate {sampling_rate} Hz")
    
    if feature_config is None:
        feature_config = FeatureConfig(sampling_rate=sampling_rate)
    
    # Set channel names based on sensor type
    if sensor_type == "current":
        channel_names = ["current_phase_a", "current_phase_b"]
    elif sensor_type == "vibration":
        channel_names = ["vibration_x", "vibration_y", "vibration_z", "vibration_w"]
    else:
        n_channels = windows.shape[2] if len(windows.shape) == 3 else 1
        channel_names = [f"ch{i}" for i in range(n_channels)]
    
    extractor = FeatureExtractor(feature_config)
    signal_features, signal_feature_names = extractor.extract_features_batch(windows, channel_names)
    
    # Extract categorical features from metadata if provided
    if metadata_list is not None:
        categorical_features, categorical_feature_names = extract_categorical_features(metadata_list)
        
        # Combine signal and categorical features
        if categorical_features.size > 0:
            feature_matrix = np.hstack([signal_features, categorical_features])
            feature_names = signal_feature_names + categorical_feature_names
        else:
            feature_matrix = signal_features
            feature_names = signal_feature_names
    else:
        feature_matrix = signal_features
        feature_names = signal_feature_names
    
    # Apply PCA if requested
    pca_reducer = None
    if feature_config.apply_pca and PCA_AVAILABLE:
        try:
            # Create PCA configuration for time-domain features
            pca_config = create_pca_config_for_time_domain(
                n_components=feature_config.pca_n_components,
                variance_threshold=feature_config.pca_variance_threshold,
                keep_original=False  # Replace time-domain features with PCA components
            )
            
            # Apply PCA
            pca_reducer = PCAFeatureReducer(pca_config)
            feature_matrix, feature_names = pca_reducer.fit_transform(feature_matrix, feature_names)
            
            # Log PCA results
            pca_summary = pca_reducer.get_pca_summary()
            if pca_summary:
                logger.info(f"PCA applied: {pca_summary['n_original_features']} -> {pca_summary['n_components']} components")
                logger.info(f"Total variance explained: {pca_summary['total_variance_explained']:.4f}")
            
        except Exception as e:
            logger.warning(f"PCA application failed: {e}. Continuing without PCA.")
    elif feature_config.apply_pca and not PCA_AVAILABLE:
        logger.warning("PCA requested but not available. Install scikit-learn for PCA functionality.")
    
    logger.info(f"Extracted {len(feature_names)} features from {len(windows)} windows")
    if metadata_list is not None:
        original_signal_count = len(signal_feature_names)
        if feature_config.apply_pca and pca_reducer is not None:
            pca_summary = pca_reducer.get_pca_summary()
            if pca_summary:
                signal_count = len([name for name in feature_names if not name.startswith('time_pca')])
                pca_count = len([name for name in feature_names if name.startswith('time_pca')])
                logger.info(f"  - Original signal features: {original_signal_count}")
                logger.info(f"  - Non-PCA signal features: {signal_count}")
                logger.info(f"  - PCA components: {pca_count}")
        else:
            logger.info(f"  - Signal features: {len(signal_feature_names)}")
        
        if metadata_list:
            categorical_count = len(feature_names) - len(signal_feature_names)
            if feature_config.apply_pca and pca_reducer is not None:
                # Adjust count after PCA
                categorical_count = len([name for name in feature_names 
                                       if name in categorical_feature_names])
            logger.info(f"  - Categorical features: {categorical_count}")
    
    return feature_matrix, feature_names