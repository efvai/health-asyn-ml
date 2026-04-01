"""
Feature extraction module for motor health monitoring data.

This module provides comprehensive feature extraction capabilities for time series data,
including time-domain, frequency-domain, and time-frequency features commonly used
in motor health monitoring and fault diagnosis.

This module now uses modular feature extraction classes for better organization and scalability.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import logging


# Import modular feature classes
from .features import (
    TimeDomainFeatures,
    FrequencyDomainFeatures,
    HilbertEnvelopeFeatures,
    FeatureConfig,
)

logger = logging.getLogger(__name__)


def _sampling_rate_key_for_sensor(sensor_type: str) -> str:
    sensor = sensor_type.lower()
    if sensor == "current":
        return "sample_rate_current_hz"
    if sensor == "vibration":
        return "sample_rate_vibro_hz"
    raise ValueError(f"Unsupported sensor_type for sampling-rate lookup: {sensor_type!r}")


def _resolve_sampling_rate_from_metadata(
    metadata: Dict[str, Any],
    *,
    sensor_type: str,
    index: int,
) -> float:
    key = _sampling_rate_key_for_sensor(sensor_type)
    if key not in metadata:
        raise ValueError(f"metadata_list[{index}] missing required key '{key}'")

    raw_value = metadata.get(key)
    try:
        sampling_rate = float(raw_value)
    except (TypeError, ValueError):
        raise ValueError(
            f"metadata_list[{index}]['{key}'] must be numeric, got {raw_value!r}"
        ) from None

    if sampling_rate <= 0:
        raise ValueError(
            f"metadata_list[{index}]['{key}'] must be > 0, got {sampling_rate!r}"
        )

    return sampling_rate
    
class FeatureExtractor:
    """Main feature extraction class."""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.time_domain = TimeDomainFeatures()
        self.frequency_domain = FrequencyDomainFeatures()
        self.hilbert_envelope = HilbertEnvelopeFeatures()

    def extract_features(
        self,
        signal: np.ndarray,
        channel_name: str = "ch",
        *,
        sampling_rate: Optional[float] = None,
    ) -> Dict[str, float]:
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
            if sampling_rate is None:
                raise ValueError(
                    "sampling_rate is required for frequency_domain features. "
                    "Resolve it from metadata before extraction."
                )
            window_type = freq_params.get("window_type", self.config.window_type)
            fft_result = self.frequency_domain.fft_features(
                signal, sampling_rate, window_type
            )
            fft_features, magnitude, freqs = fft_result
            features.update({f"{channel_name}_{k}": v for k, v in fft_features.items()})
        
        return features
    
    def extract_features_multichannel(
        self,
        signal: np.ndarray,
        channel_names: Optional[List[str]] = None,
        *,
        sampling_rate: Optional[float] = None,
    ) -> Dict[str, float]:
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
        selected_names = self.config.resolve_selected_channels(channel_names)
        channel_indices: List[int] = [
            name_to_index[name] for name in selected_names if name in name_to_index
        ]

        for idx in channel_indices:
            ch_name = channel_names[idx]
            ch_signal = signal[:, idx]
            ch_features = self.extract_features(
                ch_signal,
                ch_name,
                sampling_rate=sampling_rate,
            )
            all_features.update(ch_features)
        
        return all_features
    
    def extract_features_batch(
        self,
        windows: np.ndarray,
        channel_names: Optional[List[str]] = None,
        *,
        sampling_rates: Optional[List[float]] = None,
    ) -> tuple:
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

        if sampling_rates is not None and len(sampling_rates) != n_windows:
            raise ValueError(
                f"sampling_rates length ({len(sampling_rates)}) must match n_windows ({n_windows})"
            )

        if self.config.frequency_domain and sampling_rates is None:
            raise ValueError(
                "sampling_rates are required when frequency_domain features are enabled"
            )
        
        if channel_names is None:
            channel_names = [f"ch{i+1}" for i in range(n_channels)]

        first_sampling_rate = sampling_rates[0] if sampling_rates is not None else None
        
        # Extract features from first window to get feature names
        sample_features = self.extract_features_multichannel(
            windows[0],
            channel_names,
            sampling_rate=first_sampling_rate,
        )
        feature_names = list(sample_features.keys())
        n_features = len(feature_names)
        
        # Pre-allocate feature matrix
        feature_matrix = np.zeros((n_windows, n_features))
        
        # Extract features for all windows
        for i in range(n_windows):
            current_sampling_rate = sampling_rates[i] if sampling_rates is not None else None
            window_features = self.extract_features_multichannel(
                windows[i],
                channel_names,
                sampling_rate=current_sampling_rate,
            )
            
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

def extract_features_for_ml(windows: np.ndarray, 
                           sensor_type: str,
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

    n_windows = windows.shape[0]
    if metadata_list is not None and len(metadata_list) != n_windows:
        raise ValueError(
            f"metadata_list length ({len(metadata_list)}) must match n_windows ({n_windows})"
        )

    sampling_rates: Optional[List[float]] = None
    if feature_config.frequency_domain:
        if metadata_list is None:
            raise ValueError(
                "metadata_list is required when frequency_domain features are enabled"
            )
        sampling_rates = [
            _resolve_sampling_rate_from_metadata(meta, sensor_type=sensor_type, index=i)
            for i, meta in enumerate(metadata_list)
        ]
     
    extractor = FeatureExtractor(feature_config)
    signal_features, signal_feature_names = extractor.extract_features_batch(
        windows,
        sampling_rates=sampling_rates,
    )
    
    return signal_features, signal_feature_names