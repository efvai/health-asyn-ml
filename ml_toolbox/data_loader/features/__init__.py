"""
Feature extraction modules for motor health monitoring data.

This package provides modular feature extraction capabilities organized by feature type:
- time_domain: Statistical and time-domain features
- frequency_domain: FFT-based spectral features  
- envelope: Hilbert envelope and carrier signal analysis features

The modular design allows for easy extension and reuse of feature types.
"""

try:
    from .time_domain import TimeDomainFeatures
    from .frequency_domain import FrequencyDomainFeatures
    from .envelope import HilbertEnvelopeFeatures
    from .base import FeatureConfig, FeatureFamilyConfig
    
    __all__ = [
        'TimeDomainFeatures',
        'FrequencyDomainFeatures', 
        'HilbertEnvelopeFeatures',
        'FeatureConfig',
        'FeatureFamilyConfig'
    ]
except ImportError as e:
    # Handle potential import errors gracefully
    import logging
    logging.getLogger(__name__).warning(f"Some feature modules could not be imported: {e}")
    __all__ = []