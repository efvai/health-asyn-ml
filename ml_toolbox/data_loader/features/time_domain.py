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
        features['rms'] = np.sqrt(np.mean(signal**2))
        features['ptp'] = np.ptp(signal)
        features['skewness'] = stats.skew(signal)
        features['kurtosis'] = stats.kurtosis(signal)
    
        eps = 1e-12
        peak = np.max(np.abs(signal))
        mean_abs = np.mean(np.abs(signal))
        features['crest_factor'] = peak / (features['rms'] + eps)
        features['form_factor'] = np.sqrt(np.mean(signal**2)) / (mean_abs + eps)
        
        return features
    
    
    @staticmethod
    def cross_correlation_features(signal_a: np.ndarray, signal_b: np.ndarray, 
                                  ch1_name: str = "ch1", ch2_name: str = "ch2") -> Dict[str, float]:
        """
        Extract cross-correlation features between two time series.
        
        Args:
            signal_a: First signal
            signal_b: Second signal  
            ch1_name: Name of first channel
            ch2_name: Name of second channel
            
        Returns:
            Dictionary of cross-correlation features
        """
        features = {}
        
        # Time domain cross-correlation
        min_len = min(len(signal_a), len(signal_b))
        if min_len > 1:
            corr_coef = np.corrcoef(signal_a[:min_len], signal_b[:min_len])[0, 1]
            features[f"{ch1_name}_{ch2_name}_time_corr"] = float(corr_coef) if not np.isnan(corr_coef) else 0.0
        else:
            features[f"{ch1_name}_{ch2_name}_time_corr"] = 0.0
        
        # RMS ratio
        rms_a = np.sqrt(np.mean(signal_a**2))
        rms_b = np.sqrt(np.mean(signal_b**2))
        features[f"{ch1_name}_{ch2_name}_rms_ratio"] = float(rms_a / (rms_b + 1e-12))
        
        # Crest factor differences
        eps = 1e-12
        crest_a = np.max(np.abs(signal_a)) / (rms_a + eps)
        crest_b = np.max(np.abs(signal_b)) / (rms_b + eps)
        features[f"{ch1_name}_{ch2_name}_crest_diff"] = float(abs(crest_a - crest_b) / ((crest_a + crest_b)/2 + eps))
        
        return features