"""Simple feature list configuration for feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


TIME_FEATURES: Tuple[str, ...] = (
    "rms",
    "skewness",
    "kurtosis",
    "crest_factor",
    "form_factor",
)

FREQ_FEATURES: Tuple[str, ...] = (
    "spectral_centroid",
    "spectral_spread",
    "spectral_rolloff",
    "spectral_entropy",
    "a2_a1",
    "a3_a1",
)

ALL_FEATURES: Tuple[str, ...] = TIME_FEATURES + FREQ_FEATURES


@dataclass
class FeatureConfig:
    """Configuration for feature extraction.

    Specify exactly which features to compute using ``"{channel}_{feature}"`` strings.

    Available time-domain features:
        rms, skewness, kurtosis, crest_factor, form_factor

    Available frequency-domain features:
        spectral_centroid, spectral_spread, spectral_rolloff, spectral_entropy

    Example::

        FeatureConfig(features=["ch1_rms", "ch3_skewness", "ch2_spectral_centroid"])

    Unknown feature names or channel references to channels absent from a signal are
    reported via a printed warning and silently skipped.
    """

    features: List[str]

    def __post_init__(self) -> None:
        if not self.features:
            raise ValueError("FeatureConfig.features must contain at least one entry.")
        for item in self.features:
            if "_" not in item:
                print(
                    f"[FeatureConfig] Warning: '{item}' has no '_' separator — "
                    "expected format is '{channel}_{feature_name}'. This entry will be skipped."
                )
