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
import warnings

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
    harmonics_analysis: bool = False
    
    # Time-frequency features
    time_frequency: bool = False
    wavelet_features: bool = False
    
    # Advanced features
    entropy_features: bool = False
    complexity_features: bool = False
    
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
                (0, 50),      # Low frequency
                (50, 500),    # Motor fundamental and harmonics
                (500, 2000),  # High frequency
                (2000, 5000)  # Very high frequency (bearing defects)
            ]


class TimeDomainFeatures:
    """Extract time-domain features from signals."""
    
    @staticmethod
    def basic_statistics(signal: np.ndarray) -> Dict[str, float]:
        """Extract basic statistical features."""
        features = {}
        
        # Basic statistics
        features['mean'] = np.mean(signal)
        features['std'] = np.std(signal)
        features['var'] = np.var(signal)
        features['rms'] = np.sqrt(np.mean(signal**2))
        features['peak'] = np.max(np.abs(signal))
        features['peak_to_peak'] = np.ptp(signal)
        features['min'] = np.min(signal)
        features['max'] = np.max(signal)
        
        # Percentiles
        features['percentile_25'] = np.percentile(signal, 25)
        features['percentile_75'] = np.percentile(signal, 75)
        features['median'] = np.median(signal)
        features['iqr'] = features['percentile_75'] - features['percentile_25']
        
        return features
    
    @staticmethod
    def statistical_moments(signal: np.ndarray) -> Dict[str, float]:
        """Extract higher-order statistical moments."""
        features = {}
        
        # Standardized signal for moment calculation
        signal_std = (signal - np.mean(signal)) / (np.std(signal) + 1e-12)
        
        features['skewness'] = stats.skew(signal)
        features['kurtosis'] = stats.kurtosis(signal)
        features['moment_3'] = np.mean(signal_std**3)
        features['moment_4'] = np.mean(signal_std**4)
        features['moment_5'] = np.mean(signal_std**5)
        features['moment_6'] = np.mean(signal_std**6)
        
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
        features['impulse_factor'] = peak / (mean_abs + eps)
        features['clearance_factor'] = peak / (np.mean(np.sqrt(np.abs(signal)))**2 + eps)
        features['shape_factor'] = rms / (mean_abs + eps)
        
        return features
    
    @staticmethod
    def energy_features(signal: np.ndarray) -> Dict[str, float]:
        """Extract energy-based features."""
        features = {}
        
        features['energy'] = np.sum(signal**2)
        features['power'] = np.mean(signal**2)
        features['log_energy'] = np.log(features['energy'] + 1e-12)
        
        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)
        features['zero_crossing_rate'] = zero_crossings / len(signal)
        
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
            window_type: Window function type ('hann', 'hamming', 'blackman', 'none')
                        Default is 'hann' which provides good balance between main lobe width and side lobe suppression
        
        Returns:
            Tuple of (features_dict, fft_magnitude, fft_frequencies)
        """
        features = {}
        
        # Apply windowing to reduce spectral leakage
        windowed_signal = FrequencyDomainFeatures._apply_window(signal, window_type)
        
        # Compute FFT
        fft_vals = np.array(fft(windowed_signal))
        freqs = fftfreq(len(windowed_signal), 1/sampling_rate)
        
        # Only positive frequencies
        n_positive = len(freqs) // 2
        fft_magnitude = np.abs(fft_vals[:n_positive])
        fft_freqs = freqs[:n_positive]
        
        # Normalize FFT magnitude for motor fault detection
        total_energy = np.sum(fft_magnitude**2)
        if total_energy > 1e-12:
            # Normalize by total spectral energy
            fft_magnitude_normalized = fft_magnitude / np.sqrt(total_energy)
        else:
            fft_magnitude_normalized = fft_magnitude
        
        # Use normalized magnitude for feature extraction
        fft_magnitude = fft_magnitude_normalized
        
        # Spectral features
        features['spectral_centroid'] = np.sum(fft_freqs * fft_magnitude) / (np.sum(fft_magnitude) + 1e-12)
        features['spectral_spread'] = np.sqrt(np.sum((fft_freqs - features['spectral_centroid'])**2 * fft_magnitude) / (np.sum(fft_magnitude) + 1e-12))
        features['spectral_rolloff'] = FrequencyDomainFeatures._spectral_rolloff(fft_magnitude, fft_freqs, 0.85)
        features['spectral_flux'] = np.sum(np.diff(fft_magnitude)**2)
        
        # Peak frequency
        peak_idx = np.argmax(fft_magnitude)
        features['peak_frequency'] = fft_freqs[peak_idx]
        features['peak_magnitude'] = fft_magnitude[peak_idx]
        
        # Spectral energy (after normalization, this represents spectral concentration)
        features['spectral_energy'] = np.sum(fft_magnitude**2)
        
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
            
            if len(band_magnitude) > 0:
                features[f'{band_name}_energy'] = np.sum(band_magnitude**2)
                features[f'{band_name}_mean'] = np.mean(band_magnitude)
                features[f'{band_name}_std'] = np.std(band_magnitude)
                features[f'{band_name}_peak'] = np.max(band_magnitude)
                features[f'{band_name}_power_ratio'] = features[f'{band_name}_energy'] / (np.sum(magnitude**2) + 1e-12)
            else:
                features[f'{band_name}_energy'] = 0.0
                features[f'{band_name}_mean'] = 0.0
                features[f'{band_name}_std'] = 0.0
                features[f'{band_name}_peak'] = 0.0
                features[f'{band_name}_power_ratio'] = 0.0
        
        return features
    
    @staticmethod
    def harmonics_analysis(magnitude: np.ndarray, freqs: np.ndarray, 
                          fundamental_freq: float, num_harmonics: int = 10) -> Dict[str, float]:
        """Analyze harmonic content."""
        features = {}
        
        harmonic_magnitudes = []
        for h in range(1, num_harmonics + 1):
            harmonic_freq = h * fundamental_freq
            # Find closest frequency bin
            closest_idx = np.argmin(np.abs(freqs - harmonic_freq))
            if freqs[closest_idx] <= freqs[-1]:
                harmonic_magnitudes.append(magnitude[closest_idx])
                features[f'harmonic_{h}_magnitude'] = magnitude[closest_idx]
            else:
                harmonic_magnitudes.append(0.0)
                features[f'harmonic_{h}_magnitude'] = 0.0
        
        # THD (Total Harmonic Distortion)
        if len(harmonic_magnitudes) > 1:
            fundamental_mag = harmonic_magnitudes[0]
            harmonics_mag = harmonic_magnitudes[1:]
            features['thd'] = np.sqrt(np.sum(np.array(harmonics_mag)**2)) / (fundamental_mag + 1e-12)
        else:
            features['thd'] = 0.0
        
        return features


class AdvancedFeatures:
    """Extract advanced features for complex analysis."""
    
    @staticmethod
    def entropy_features(signal: np.ndarray) -> Dict[str, float]:
        """Extract entropy-based features."""
        features = {}
        
        # Skip expensive entropy calculations for long signals
        if len(signal) > 10000:
            # Use spectral entropy only for long signals
            try:
                fft_vals = np.abs(np.array(fft(signal)))
                psd = fft_vals**2
                psd_norm = psd / (np.sum(psd) + 1e-12)
                psd_norm = psd_norm[psd_norm > 1e-12]  # Remove near-zeros
                features['spectral_entropy'] = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
            except:
                features['spectral_entropy'] = 0.0
            
            # Set others to zero for speed
            features['approximate_entropy'] = 0.0
            features['sample_entropy'] = 0.0
        else:
            # For shorter signals, compute all entropy features
            try:
                features['approximate_entropy'] = AdvancedFeatures._approximate_entropy(signal[:2000])  # Limit length
            except:
                features['approximate_entropy'] = 0.0
            
            try:
                features['sample_entropy'] = AdvancedFeatures._sample_entropy(signal[:2000])  # Limit length
            except:
                features['sample_entropy'] = 0.0
            
            try:
                fft_vals = np.abs(np.array(fft(signal)))
                psd = fft_vals**2
                psd_norm = psd / (np.sum(psd) + 1e-12)
                psd_norm = psd_norm[psd_norm > 1e-12]
                features['spectral_entropy'] = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
            except:
                features['spectral_entropy'] = 0.0
        
        return features
    
    @staticmethod
    def _approximate_entropy(signal: np.ndarray, m: int = 2, r: Optional[float] = None) -> float:
        """Calculate approximate entropy."""
        if r is None:
            r = float(0.2 * np.std(signal))
        
        def _maxdist(xi, xj, m):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _phi(m):
            patterns = np.array([signal[i:i + m] for i in range(len(signal) - m + 1)])
            C = np.zeros(len(patterns))
            for i in range(len(patterns)):
                template = patterns[i]
                C[i] = sum([1 for j in range(len(patterns)) 
                           if _maxdist(template, patterns[j], m) <= r]) / float(len(patterns))
            phi = sum([np.log(c) for c in C if c > 0]) / float(len(patterns))
            return phi
        
        return _phi(m) - _phi(m + 1)
    
    @staticmethod
    def _sample_entropy(signal: np.ndarray, m: int = 2, r: Optional[float] = None) -> float:
        """Calculate sample entropy."""
        if r is None:
            r = float(0.2 * np.std(signal))
        
        def _maxdist(xi, xj):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _phi(m):
            patterns = np.array([signal[i:i + m] for i in range(len(signal) - m + 1)])
            C = 0
            for i in range(len(patterns) - 1):
                template = patterns[i]
                C += sum([1 for j in range(i + 1, len(patterns)) 
                         if _maxdist(template, patterns[j]) <= r])
            return C
        
        A = _phi(m)
        B = _phi(m + 1)
        return -np.log(B / A) if A > 0 and B > 0 else 0.0


class FeatureExtractor:
    """Main feature extraction class."""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.time_domain = TimeDomainFeatures()
        self.frequency_domain = FrequencyDomainFeatures()
        self.advanced = AdvancedFeatures()
    
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
            
            energy_features = self.time_domain.energy_features(signal)
            features.update({f"{channel_name}_{k}": v for k, v in energy_features.items()})
        
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
        
        # Advanced features
        if self.config.entropy_features:
            entropy_features = self.advanced.entropy_features(signal)
            features.update({f"{channel_name}_{k}": v for k, v in entropy_features.items()})
        
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
        for ch_idx, ch_name in enumerate(channel_names):
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
                corr = np.corrcoef(signal[:, i], signal[:, j])[0, 1]
                features[f"{ch1_name}_{ch2_name}_correlation"] = corr if not np.isnan(corr) else 0.0
                
                # Phase difference (simplified)
                fft1 = np.array(fft(signal[:, i]))
                fft2 = np.array(fft(signal[:, j]))
                phase_diff = float(np.mean(np.angle(fft1) - np.angle(fft2)))
                features[f"{ch1_name}_{ch2_name}_phase_diff"] = phase_diff
        
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
            feature_matrix[i, :] = [window_features[name] for name in feature_names]
            
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
        Tuple of (feature_matrix, feature_names)
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
    
    logger.info(f"Extracted {len(feature_names)} features from {len(windows)} windows")
    if metadata_list is not None:
        logger.info(f"  - Signal features: {len(signal_feature_names)}")
        if metadata_list:
            categorical_count = len(feature_names) - len(signal_feature_names)
            logger.info(f"  - Categorical features: {categorical_count}")
    
    return feature_matrix, feature_names