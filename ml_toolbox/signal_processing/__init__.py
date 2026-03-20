"""
Signal processing utilities for ML toolbox.

This module contains general signal processing functions that can be used
across different analysis tasks.
"""

from .spectrum_analysis import (
    compute_fft_spectrum,
    find_spectral_peaks,
    prepare_time_frequency_view,
)

__all__ = [
    'compute_fft_spectrum',
    'find_spectral_peaks',
    'prepare_time_frequency_view',
]