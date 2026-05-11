"""
Feature extraction module for motor health monitoring data.

This module provides feature extraction capabilities for time series data,
including time-domain and frequency-domain features commonly used
in motor health monitoring and fault diagnosis.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import logging


# Import modular feature classes
from .features import (
    TimeDomainFeatures,
    FrequencyDomainFeatures,
    FeatureConfig,
    TIME_FEATURES,
    FREQ_FEATURES,
)

logger = logging.getLogger(__name__)

# All recognized feature names per family (derived from the canonical constants).
KNOWN_TIME_FEATURES: frozenset = frozenset(TIME_FEATURES)
KNOWN_FREQ_FEATURES: frozenset = frozenset(FREQ_FEATURES)
KNOWN_FEATURES: frozenset = KNOWN_TIME_FEATURES | KNOWN_FREQ_FEATURES


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

        # Parse and validate feature specs at construction time.
        # Each valid entry maps channel_name -> set of feature names.
        self._channel_features: Dict[str, List[str]] = {}
        for item in config.features:
            if "_" not in item:
                # Already warned in FeatureConfig.__post_init__; skip here.
                continue
            channel, feature = item.split("_", 1)
            if feature not in KNOWN_FEATURES:
                print(
                    f"[FeatureExtractor] Warning: unknown feature '{feature}' in '{item}'. "
                    f"Known features: {sorted(KNOWN_FEATURES)}. This entry will be skipped."
                )
                continue
            self._channel_features.setdefault(channel, [])
            if feature not in self._channel_features[channel]:
                self._channel_features[channel].append(feature)

        self._needs_freq: bool = any(
            f in KNOWN_FREQ_FEATURES
            for features in self._channel_features.values()
            for f in features
        )

    def extract_features(
        self,
        signal: np.ndarray,
        channel_name: str = "ch",
        *,
        sampling_rate: Optional[float] = None,
    ) -> Dict[str, float]:
        """Extract the requested features from a single-channel signal."""
        requested = self._channel_features.get(channel_name, [])
        if not requested:
            return {}

        features: Dict[str, float] = {}

        time_requested = [f for f in requested if f in KNOWN_TIME_FEATURES]
        if time_requested:
            all_time = self.time_domain.basic_statistics(signal)
            for fname in time_requested:
                if fname in all_time:
                    features[f"{channel_name}_{fname}"] = all_time[fname]

        freq_requested = [f for f in requested if f in KNOWN_FREQ_FEATURES]
        if freq_requested:
            if sampling_rate is None:
                raise ValueError(
                    "sampling_rate is required for frequency-domain features. "
                    "Resolve it from metadata before extraction."
                )
            fft_result, _, _ = self.frequency_domain.fft_features(signal, sampling_rate)
            for fname in freq_requested:
                if fname in fft_result:
                    features[f"{channel_name}_{fname}"] = fft_result[fname]

        return features

    def extract_features_multichannel(
        self,
        signal: np.ndarray,
        channel_names: Optional[List[str]] = None,
        *,
        sampling_rate: Optional[float] = None,
    ) -> Dict[str, float]:
        """Extract features from multi-channel signal (samples × channels)."""
        if signal.ndim != 2:
            raise ValueError("Signal must be 2D (samples, channels)")

        n_samples, n_channels = signal.shape

        if channel_names is None:
            channel_names = [f"ch{i + 1}" for i in range(n_channels)]

        name_to_index = {name: idx for idx, name in enumerate(channel_names)}
        all_features: Dict[str, float] = {}

        for ch_name in self._channel_features:
            if ch_name not in name_to_index:
                print(
                    f"[FeatureExtractor] Warning: channel '{ch_name}' not found in signal "
                    f"(available: {channel_names}). Skipping."
                )
                continue
            idx = name_to_index[ch_name]
            ch_features = self.extract_features(
                signal[:, idx], ch_name, sampling_rate=sampling_rate
            )
            all_features.update(ch_features)

        return all_features

    def extract_features_batch(
        self,
        windows: np.ndarray,
        channel_names: Optional[List[str]] = None,
        *,
        sampling_rates: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract features from a batch of windows (n_windows × window_size × n_channels).

        Returns:
            (feature_matrix, feature_names) — shapes (n_windows, n_features) and (n_features,).
        """
        if windows.ndim != 3:
            raise ValueError("Windows must be 3D (n_windows, window_size, n_channels)")

        n_windows, window_size, n_channels = windows.shape

        if sampling_rates is not None and len(sampling_rates) != n_windows:
            raise ValueError(
                f"sampling_rates length ({len(sampling_rates)}) must match n_windows ({n_windows})"
            )

        if self._needs_freq and sampling_rates is None:
            raise ValueError(
                "sampling_rates are required when frequency-domain features are requested"
            )

        if channel_names is None:
            channel_names = [f"ch{i + 1}" for i in range(n_channels)]

        first_sr = sampling_rates[0] if sampling_rates is not None else None
        sample_features = self.extract_features_multichannel(
            windows[0], channel_names, sampling_rate=first_sr
        )
        feature_names = list(sample_features.keys())
        n_features = len(feature_names)

        feature_matrix = np.zeros((n_windows, n_features))
        feature_matrix[0] = list(sample_features.values())

        for i in range(1, n_windows):
            sr = sampling_rates[i] if sampling_rates is not None else None
            window_features = self.extract_features_multichannel(
                windows[i], channel_names, sampling_rate=sr
            )
            for j, name in enumerate(feature_names):
                feature_matrix[i, j] = window_features.get(name, 0.0)

        return feature_matrix, feature_names

def extract_features_for_ml(
    windows: np.ndarray,
    sensor_type: str,
    feature_config: Optional[FeatureConfig] = None,
    metadata_list: Optional[List[Dict]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Convenience function to extract features ready for ML.

    Args:
        windows: 3D array (n_windows, window_size, n_channels)
        sensor_type: Type of sensor — used only for sampling-rate metadata lookup
        feature_config: Feature configuration (required)
        metadata_list: List of metadata dicts; required when any frequency feature is requested

    Returns:
        Tuple of (feature_matrix, feature_names)
    """
    if feature_config is None:
        raise ValueError(
            "feature_config is required. "
            "Example: FeatureConfig(features=['ch1_rms', 'ch2_skewness'])"
        )

    n_windows = windows.shape[0]
    if metadata_list is not None and len(metadata_list) != n_windows:
        raise ValueError(
            f"metadata_list length ({len(metadata_list)}) must match n_windows ({n_windows})"
        )

    extractor = FeatureExtractor(feature_config)

    sampling_rates: Optional[List[float]] = None
    if extractor._needs_freq:
        if metadata_list is None:
            raise ValueError(
                "metadata_list is required when frequency-domain features are requested"
            )
        sampling_rates = [
            _resolve_sampling_rate_from_metadata(meta, sensor_type=sensor_type, index=i)
            for i, meta in enumerate(metadata_list)
        ]

    return extractor.extract_features_batch(
        windows,
        sampling_rates=sampling_rates,
    )