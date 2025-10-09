"""
Data loading utilities for the ML toolbox.
"""

from .dataset_manager import DatasetManager, DatasetInfo
from .data_loader import DataLoader  
from .config import ConfigManager, DatasetConfig, SensorConfig
from .windowing import (
    WindowConfig, WindowExtractor, StratifiedWindowExtractor, 
    SlidingWindowGenerator, WindowAnalyzer, create_windows_for_ml
)
from .feature_extraction import (
    FeatureConfig, FeatureExtractor, TimeDomainFeatures, 
    FrequencyDomainFeatures, extract_features_for_ml, CURRENT_SAMPLING_RATE, VIBRATION_SAMPLING_RATE
)

from .envelope_analyzer import (
    HilbertEnvelopeAnalyzer, EnvelopeConfig
)

# Try to import PCA functionality - may not be available if scikit-learn not installed
try:
    from .pca_reduction import (
        PCAConfig, PCAFeatureReducer, apply_pca_to_features, 
        create_pca_config_for_time_domain
    )
    _PCA_AVAILABLE = True
except ImportError:
    _PCA_AVAILABLE = False

__all__ = [
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
    'CURRENT_SAMPLING_RATE',
    'VIBRATION_SAMPLING_RATE',
    'HilbertEnvelopeAnalyzer',
    'EnvelopeConfig'
]

# Add PCA exports if available
if _PCA_AVAILABLE:
    __all__.extend([
        'PCAConfig',
        'PCAFeatureReducer', 
        'apply_pca_to_features',
        'create_pca_config_for_time_domain'
    ])