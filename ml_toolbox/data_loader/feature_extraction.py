"""
Feature extraction module for motor health monitoring data.

This module provides comprehensive feature extraction capabilities for time series data,
including time-domain, frequency-domain, and time-frequency features commonly used
in motor health monitoring and fault diagnosis.

This module now uses modular feature extraction classes for better organization and scalability.
"""

import numpy as np
from itertools import combinations
from typing import Dict, List, Optional, Union, Callable, Tuple
import logging


# Import modular feature classes
from .features import (
    TimeDomainFeatures,
    FrequencyDomainFeatures,
    HilbertEnvelopeFeatures,
    FeatureConfig,
)

logger = logging.getLogger(__name__)
    
class FeatureExtractor:
    """Main feature extraction class."""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.time_domain = TimeDomainFeatures()
        self.frequency_domain = FrequencyDomainFeatures()
        self.hilbert_envelope = HilbertEnvelopeFeatures()

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

        if self.config.hilbert_envelope:
            hilbert_params = self.config.get_params("hilbert_envelope")
            allowed_keys = {"bandpass_low", "bandpass_high", "expected_carrier", "carrier_bandwidth"}
            hilbert_kwargs = {
                key: hilbert_params[key] for key in allowed_keys if key in hilbert_params
            }
            hilbert_features = self.hilbert_envelope.hilbert_envelope_features(
                signal,
                **hilbert_kwargs
            )
            features.update({f"{channel_name}_{k}": v for k, v in hilbert_features.items()})

        # Frequency domain features
        if self.config.frequency_domain:
            freq_params = self.config.get_params("frequency_domain")
            sampling_rate = freq_params.get("sampling_rate", self.config.sampling_rate)
            window_type = freq_params.get("window_type", self.config.window_type)
            fft_result = self.frequency_domain.fft_features(
                signal, sampling_rate, window_type
            )
            fft_features, magnitude, freqs = fft_result
            features.update({f"{channel_name}_{k}": v for k, v in fft_features.items()})
        
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
        
        # Extract features for requested channels
        name_to_index = {name: idx for idx, name in enumerate(channel_names)}
        selected_names = self.config.resolve_channel_scope(channel_names)
        channel_indices: List[int] = [
            name_to_index[name] for name in selected_names if name in name_to_index
        ]

        if not channel_indices:
            channel_indices = list(range(n_channels))
            selected_names = [channel_names[idx] for idx in channel_indices]

        for idx in channel_indices:
            ch_name = channel_names[idx]
            ch_signal = signal[:, idx]
            ch_features = self.extract_features(ch_signal, ch_name)
            all_features.update(ch_features)
        
        # Cross-channel features: iterate channel pairs once and call
        # enabled cross-channel extractors conditionally to avoid repetition.
        if self.config.cross_channel and len(channel_indices) > 1:
            cross_params = self.config.get_params("cross_channel")
            requested_pairs = cross_params.get("pairs") if isinstance(cross_params, dict) else None

            cross_time_enabled = True
            cross_env_enabled = True

            if isinstance(cross_params, dict):
                cross_time_enabled = cross_params.get("time_domain", self.config.time_domain)
                cross_env_enabled = cross_params.get(
                    "hilbert_envelope", self.config.hilbert_envelope
                )

            pair_indices = self._resolve_channel_pairs(
                requested_pairs,
                channel_indices,
                channel_names,
            )

            for i_idx, j_idx in pair_indices:
                ch1_name = channel_names[i_idx]
                ch2_name = channel_names[j_idx]
                ch1_signal = signal[:, i_idx]
                ch2_signal = signal[:, j_idx]

                if cross_time_enabled and self.config.time_domain:
                    cc_feats = self.time_domain.cross_correlation_features(
                        ch1_signal, ch2_signal, ch1_name, ch2_name
                    )
                    all_features.update(cc_feats)

                if cross_env_enabled and self.config.hilbert_envelope:
                    env_cross_features = self.hilbert_envelope.hilbert_envelope_cross_features(
                        ch1_signal,
                        ch2_signal,
                        ch1_name,
                        ch2_name,
                    )
                    all_features.update(env_cross_features)
        
        return all_features
    
    # NOTE: cross-channel features are now provided by `TimeDomainFeatures.cross_correlation_features`
    
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
            
        return feature_matrix, feature_names

    @staticmethod
    def _resolve_channel_pairs(
        requested_pairs,
        default_indices: List[int],
        channel_names: List[str],
    ) -> List[Tuple[int, int]]:
        """Resolve channel pair definitions to index tuples."""

        if not requested_pairs:
            return list(combinations(default_indices, 2))

        name_to_index = {name: idx for idx, name in enumerate(channel_names)}
        resolved: List[Tuple[int, int]] = []
        seen = set()

        for pair in requested_pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            first, second = pair

            if isinstance(first, str):
                idx_a = name_to_index.get(first)
            elif isinstance(first, int):
                idx_a = first
            else:
                idx_a = None

            if isinstance(second, str):
                idx_b = name_to_index.get(second)
            elif isinstance(second, int):
                idx_b = second
            else:
                idx_b = None

            if idx_a is None or idx_b is None or idx_a == idx_b:
                continue

            pair_key: Tuple[int, int] = (min(idx_a, idx_b), max(idx_a, idx_b))
            if pair_key in seen:
                continue

            seen.add(pair_key)
            resolved.append(pair_key)

        return resolved if resolved else list(combinations(default_indices, 2))


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
        
        # Check frequency_dir for implicit load condition (e.g., 20hz_4, 20hz_5, 20hz_6)
        freq_dir = meta.get('frequency_dir', '')
        import re
        freq_match = re.search(r'(\d+)hz[_\s](\d+)', freq_dir)
        
        # If frequency_dir number > 3, it's under load
        if freq_match and int(freq_match.group(2)) > 3:
            load_val = 1
        else:
            # Otherwise use the explicit load field
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
                           sensor_type: str = "current",
                           feature_config: Optional[FeatureConfig] = None,
                           metadata_list: Optional[List[Dict]] = None) -> tuple:
    """
    Convenience function to extract features ready for ML.
    
    Args:
        windows: 3D array (n_windows, window_size, n_channels)
        sensor_type: Type of sensor ("current" or "vibration")
        feature_config: Custom feature configuration
        metadata_list: Optional list of metadata dicts for categorical features
        
    Returns:
        Tuple of (feature_matrix, feature_names)
    """
    if feature_config is None:
        feature_config = FeatureConfig.for_sensor(sensor_type)
    else:
        feature_config = feature_config.copy()
        feature_config.apply_sensor_profile(sensor_type, override=False)
    
    # Set channel names based on sensor type
    if sensor_type == "current":
        channel_names = ["ph_a", "ph_b"]
    elif sensor_type == "vibration":
        channel_names = ["v_ch1_x", "v_ch2_z", "v_ch3_x", "v_ch4_z"]
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
    
    return feature_matrix, feature_names