"""
Signal processing utilities for ML toolbox.

This module contains general signal processing functions that can be used
across different analysis tasks.
"""

from .peak_finder import find_spectral_peaks, find_dominant_frequency, find_harmonics

__all__ = [
    "find_spectral_peaks",
    "find_dominant_frequency",
    "find_harmonics",
]