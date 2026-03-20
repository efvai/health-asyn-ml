"""
ML Toolbox for Health Monitoring and Sensor Data Analysis.

This package provides tools for reading sensor data and managing dataset metadata.
"""

# Import data_io submodule for backward compatibility
from . import data_io

# Import new dataset management classes from data_loader subpackage
from .data_loader.dataset_manager import DatasetManager
from .data_loader.data_loader import DataLoader

# Import windowing utilities
from .data_loader.windowing import (
    WindowConfig, WindowExtractor, StratifiedWindowExtractor, 
    create_windows_for_ml
)

# Import feature extraction utilities
from .data_loader.feature_extraction import (
    FeatureExtractor, extract_features_for_ml, extract_categorical_features
)
from .data_loader.features import (
    FeatureConfig, TimeDomainFeatures, FrequencyDomainFeatures, HilbertEnvelopeFeatures
)

from .data_loader.envelope_analyzer import (
    HilbertEnvelopeAnalyzer, EnvelopeConfig
)

# Import analysis modules
from . import analysis

# Import signal processing utilities
from . import signal_processing
from .signal_processing import compute_fft_spectrum, find_spectral_peaks

__all__ = [
    'data_io',
    'analysis',
    'signal_processing',
    'DatasetManager', 
    'DataLoader',
    'WindowConfig',
    'WindowExtractor',
    'StratifiedWindowExtractor',
    'create_windows_for_ml',
    'FeatureConfig',
    'FeatureExtractor',
    'TimeDomainFeatures',
    'FrequencyDomainFeatures',
    'HilbertEnvelopeFeatures',
    'extract_features_for_ml',
    'extract_categorical_features',
    'HilbertEnvelopeAnalyzer',
    'EnvelopeConfig',
    'compute_fft_spectrum',
    'find_spectral_peaks'
]

__version__ = "1.0.0"
