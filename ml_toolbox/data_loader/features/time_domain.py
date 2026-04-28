"""
Time-domain feature extraction for motor health monitoring data.

This module provides statistical and time-domain features commonly used
in motor health monitoring and fault diagnosis.
"""

import numpy as np
from typing import Dict
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class TimeDomainFeatures:
    """Extract time-domain features from signals."""
    
    @staticmethod
    def basic_statistics(signal: np.ndarray) -> Dict[str, float]:
        """Extract basic statistical features."""
        features = {}
        
        # Basic statistics
        #features['rms'] = np.sqrt(np.mean(signal**2))
        #features['ptp'] = np.ptp(signal)
        features['skewness'] = stats.skew(signal)
        features['kurtosis'] = stats.kurtosis(signal)
    
        eps = 1e-12
        peak = np.max(np.abs(signal))
        mean_abs = np.mean(np.abs(signal))
        rms = np.sqrt(np.mean(signal**2))
        features['crest_factor'] = peak / (rms + eps)
        features['form_factor'] = rms / (mean_abs + eps)
        
        return features
    