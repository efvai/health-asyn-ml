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
    FeatureExtractor, extract_features_for_ml
)
from .data_loader.features import (
    FeatureConfig, TimeDomainFeatures, FrequencyDomainFeatures
)

# Import analysis modules
from . import analysis

# Import signal processing utilities
from . import signal_processing

# Import preprocessing utilities
from . import preprocessing
from .preprocessing import ButterworthLPF

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
    'extract_features_for_ml',
    'preprocessing',
    'ButterworthLPF',
]

__version__ = "1.0.0"
