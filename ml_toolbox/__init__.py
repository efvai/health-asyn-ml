"""
ML Toolbox for Health Monitoring and Sensor Data Analysis.

This package provides tools for reading sensor data and managing dataset metadata.
"""

# Import data_io submodule for backward compatibility
from . import data_io

# Import new dataset management classes from data_loader subpackage
from .data_loader.dataset_manager import DatasetManager, DatasetInfo
from .data_loader.data_loader import DataLoader
from .data_loader.config import ConfigManager, DatasetConfig, SensorConfig

# Import windowing utilities
from .data_loader.windowing import (
    WindowConfig, WindowExtractor, StratifiedWindowExtractor, 
    SlidingWindowGenerator, WindowAnalyzer, create_windows_for_ml
)

# Import feature extraction utilities
from .data_loader.feature_extraction import (
    FeatureConfig, FeatureExtractor, TimeDomainFeatures, 
    FrequencyDomainFeatures, extract_features_for_ml,
    extract_categorical_features
)

# Import analysis modules
from . import analysis

# Import key analysis functions for convenience
from .analysis import (
    evaluate_incremental_features_cv,
    plot_incremental_feature_performance,
    write_incremental_results_to_excel,
    extract_features_for_frequency,
    run_comprehensive_frequency_analysis
)

__all__ = [
    'data_io',
    'analysis',
    'DatasetManager', 
    'DatasetInfo',
    'DataLoader',
    'ConfigManager', 
    'DatasetConfig', 
    'SensorConfig',
    'WindowConfig',
    'WindowExtractor',
    'StratifiedWindowExtractor',
    'SlidingWindowGenerator', 
    'WindowAnalyzer',
    'create_windows_for_ml',
    'FeatureConfig',
    'FeatureExtractor',
    'TimeDomainFeatures',
    'FrequencyDomainFeatures',
    'extract_features_for_ml',
    'extract_categorical_features',
    'evaluate_incremental_features_cv',
    'plot_incremental_feature_performance',
    'write_incremental_results_to_excel',
    'extract_features_for_frequency',
    'run_comprehensive_frequency_analysis',
]

__version__ = "1.0.0"
