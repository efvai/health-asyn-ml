"""
PCA (Principal Component Analysis) feature reduction module for motor health monitoring.

This module provides PCA-based dimensionality reduction for the extracted features,
specifically targeting time-domain features including statistical moments and shape factors.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
import os

logger = logging.getLogger(__name__)


@dataclass
class PCAConfig:
    """Configuration for PCA feature reduction."""
    # General PCA settings
    n_components: Optional[int] = None  # Number of components (None for auto-selection)
    variance_threshold: float = 0.95  # Cumulative variance threshold for auto-selection
    
    # Feature selection for PCA
    apply_to_time_domain: bool = True
    apply_to_statistical_moments: bool = True
    apply_to_shape_factors: bool = True
    apply_to_frequency_domain: bool = False
    apply_to_spectral_features: bool = False
    apply_to_hht_features: bool = False
    
    # Preprocessing
    standardize_features: bool = True  # Standardize features before PCA
    
    # Output settings
    keep_original_features: bool = False  # Keep original features alongside PCA components
    pca_prefix: str = "pca"  # Prefix for PCA component names
    

class PCAFeatureReducer:
    """PCA-based feature reduction for motor health monitoring features."""
    
    def __init__(self, config: PCAConfig):
        """
        Initialize PCA feature reducer.
        
        Args:
            config: PCA configuration
        """
        self.config = config
        self.pca_model = None
        self.scaler = None
        self.feature_names_original = None
        self.feature_names_selected = None
        self.feature_names_pca = None
        self.feature_mask = None
        self.is_fitted = False
        
        # Initialize scaler if standardization is enabled
        if self.config.standardize_features:
            self.scaler = StandardScaler()
    
    def _select_features_for_pca(self, feature_names: List[str]) -> np.ndarray:
        """
        Select which features to apply PCA to based on configuration.
        
        Args:
            feature_names: List of all feature names
            
        Returns:
            Boolean mask indicating which features to include in PCA
        """
        mask = np.zeros(len(feature_names), dtype=bool)
        
        for i, name in enumerate(feature_names):
            # Skip categorical features explicitly
            if any(cat_feature in name for cat_feature in [
                'frequency_hz', 'load_', 'sensor_'
            ]):
                continue
            
            # Skip frequency domain features explicitly to avoid false positives
            if any(freq_feature in name for freq_feature in [
                'spectral_', 'peak_frequency', 'peak_magnitude', 'band_', 'harmonic_', 'thd'
            ]):
                # Only include if explicitly requested for frequency domain
                if self.config.apply_to_frequency_domain or self.config.apply_to_spectral_features:
                    mask[i] = True
                continue
            
            # Skip HHT features explicitly
            if any(hht_feature in name for hht_feature in [
                'imf_', 'hht_', 'total_imf_energy', 'energy_distribution_entropy',
                'dominant_imf_index', 'instantaneous_bandwidth'
            ]):
                if self.config.apply_to_hht_features:
                    mask[i] = True
                continue
            
            # Check time domain features (basic statistics)
            if self.config.apply_to_time_domain:
                # More specific patterns that won't match frequency features
                if any(td_feature == name.split('_')[-1] or f'_{td_feature}' in name for td_feature in [
                    'mean', 'std', 'var', 'rms', 'peak', 'min', 'max', 'median', 'iqr',
                    'energy', 'power'
                ]) and not any(exclude in name for exclude in ['spectral_', 'band_', 'harmonic_']):
                    mask[i] = True
                    continue
                
                # Specific time-domain patterns
                if any(pattern in name for pattern in [
                    'peak_to_peak', 'percentile_', 'log_energy', 'zero_crossing_rate'
                ]):
                    mask[i] = True
                    continue
            
            # Check statistical moments
            if self.config.apply_to_statistical_moments:
                if any(stat_feature in name for stat_feature in [
                    'skewness', 'kurtosis', 'moment_3', 'moment_4', 'moment_5', 'moment_6'
                ]):
                    mask[i] = True
                    continue
            
            # Check shape factors
            if self.config.apply_to_shape_factors:
                if any(shape_feature in name for shape_feature in [
                    'crest_factor', 'form_factor', 'impulse_factor', 
                    'clearance_factor', 'shape_factor'
                ]):
                    mask[i] = True
                    continue
        
        logger.info(f"Selected {np.sum(mask)} out of {len(feature_names)} features for PCA")
        return mask
    
    def fit(self, feature_matrix: np.ndarray, feature_names: List[str]) -> 'PCAFeatureReducer':
        """
        Fit PCA model on the feature matrix.
        
        Args:
            feature_matrix: 2D array (n_samples, n_features)
            feature_names: List of feature names
            
        Returns:
            Self for method chaining
        """
        self.feature_names_original = feature_names.copy()
        
        # Select features for PCA
        self.feature_mask = self._select_features_for_pca(feature_names)
        selected_features = feature_matrix[:, self.feature_mask]
        self.feature_names_selected = [name for i, name in enumerate(feature_names) if self.feature_mask[i]]
        
        if selected_features.shape[1] == 0:
            logger.warning("No features selected for PCA. Check configuration.")
            self.is_fitted = False
            return self
        
        logger.info(f"Applying PCA to {selected_features.shape[1]} selected features")
        
        # Standardize features if enabled
        if self.config.standardize_features and self.scaler is not None:
            selected_features = self.scaler.fit_transform(selected_features)
        
        # Determine number of components
        n_components = self._determine_n_components(selected_features)
        
        # Fit PCA
        self.pca_model = PCA(n_components=n_components, random_state=42)
        self.pca_model.fit(selected_features)
        
        # Generate PCA feature names
        self.feature_names_pca = [f"{self.config.pca_prefix}_comp_{i+1}" for i in range(n_components)]
        
        # Log PCA results
        explained_variance_ratio = self.pca_model.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        logger.info(f"PCA fitted with {n_components} components")
        logger.info(f"Explained variance ratio: {explained_variance_ratio[:5]}...")  # Show first 5
        logger.info(f"Cumulative variance explained: {cumulative_variance[-1]:.4f}")
        
        self.is_fitted = True
        
        return self
    
    def _determine_n_components(self, features: np.ndarray) -> int:
        """
        Determine the optimal number of PCA components.
        
        Args:
            features: Feature matrix to analyze
            
        Returns:
            Number of components to use
        """
        if self.config.n_components is not None:
            return min(self.config.n_components, features.shape[1], features.shape[0])
        
        # Use variance threshold to determine components
        temp_pca = PCA()
        temp_pca.fit(features)
        
        cumulative_variance = np.cumsum(temp_pca.explained_variance_ratio_)
        n_components = int(np.argmax(cumulative_variance >= self.config.variance_threshold) + 1)
        
        # Ensure we don't exceed matrix dimensions
        n_components = min(n_components, features.shape[1], features.shape[0])
        
        logger.info(f"Auto-selected {n_components} components for {self.config.variance_threshold:.2%} variance")
        return n_components
    
    def transform(self, feature_matrix: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Transform features using fitted PCA model.
        
        Args:
            feature_matrix: 2D array (n_samples, n_features)
            
        Returns:
            Tuple of (transformed_features, feature_names)
        """
        if not self.is_fitted:
            raise ValueError("PCA model not fitted. Call fit() first.")
        
        if (self.feature_mask is None or self.pca_model is None or 
            self.feature_names_original is None or self.feature_names_pca is None):
            logger.warning("No PCA transformation available")
            return feature_matrix, self.feature_names_original or []
        
        # Select features for PCA
        selected_features = feature_matrix[:, self.feature_mask]
        
        if selected_features.shape[1] == 0:
            logger.warning("No features selected for transformation")
            return feature_matrix, self.feature_names_original
        
        # Standardize features if enabled
        if self.config.standardize_features and self.scaler is not None:
            selected_features = self.scaler.transform(selected_features)
        
        # Apply PCA transformation
        pca_features = self.pca_model.transform(selected_features)
        
        # Combine results based on configuration
        if self.config.keep_original_features:
            # Keep all original features and add PCA components
            combined_features = np.hstack([feature_matrix, pca_features])
            combined_names = self.feature_names_original + self.feature_names_pca
        else:
            # Replace selected features with PCA components, keep non-selected features
            non_selected_features = feature_matrix[:, ~self.feature_mask]
            non_selected_names = [name for i, name in enumerate(self.feature_names_original) if not self.feature_mask[i]]
            
            if non_selected_features.shape[1] > 0:
                combined_features = np.hstack([non_selected_features, pca_features])
                combined_names = non_selected_names + self.feature_names_pca
            else:
                combined_features = pca_features
                combined_names = self.feature_names_pca
        
        logger.info(f"PCA transformation: {feature_matrix.shape[1]} -> {combined_features.shape[1]} features")
        return combined_features, combined_names
    
    def fit_transform(self, feature_matrix: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Fit PCA model and transform features in one step.
        
        Args:
            feature_matrix: 2D array (n_samples, n_features)
            feature_names: List of feature names
            
        Returns:
            Tuple of (transformed_features, feature_names)
        """
        self.fit(feature_matrix, feature_names)
        return self.transform(feature_matrix)
    
    def get_feature_importance(self) -> Optional[Dict]:
        """
        Get feature importance based on PCA loadings.
        
        Returns:
            Dictionary with component loadings and feature contributions
        """
        if not self.is_fitted or self.pca_model is None:
            return None
        
        # Get PCA components (loadings)
        components = self.pca_model.components_
        
        # Calculate feature importance as sum of absolute loadings
        feature_importance = np.sum(np.abs(components), axis=0)
        
        # Normalize to sum to 1
        feature_importance = feature_importance / np.sum(feature_importance)
        
        return {
            'feature_names': self.feature_names_selected,
            'importance_scores': feature_importance,
            'pca_components': components,
            'explained_variance_ratio': self.pca_model.explained_variance_ratio_
        }
    
    def get_pca_summary(self) -> Optional[Dict]:
        """
        Get summary of PCA model performance.
        
        Returns:
            Dictionary with PCA summary statistics
        """
        if not self.is_fitted or self.pca_model is None:
            return None
        
        explained_variance = self.pca_model.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        return {
            'n_components': len(explained_variance),
            'n_original_features': len(self.feature_names_selected) if self.feature_names_selected else 0,
            'explained_variance_ratio': explained_variance.tolist(),
            'cumulative_variance': cumulative_variance.tolist(),
            'total_variance_explained': float(cumulative_variance[-1]),
            'feature_reduction_ratio': len(explained_variance) / max(1, len(self.feature_names_selected or [])),
            'selected_feature_names': self.feature_names_selected,
            'pca_feature_names': self.feature_names_pca
        }


def apply_pca_to_features(feature_matrix: np.ndarray, 
                         feature_names: List[str],
                         pca_config: Optional[PCAConfig] = None) -> Tuple[np.ndarray, List[str], PCAFeatureReducer]:
    """
    Convenience function to apply PCA to feature matrix.
    
    Args:
        feature_matrix: 2D array (n_samples, n_features)
        feature_names: List of feature names
        pca_config: PCA configuration (uses default if None)
        
    Returns:
        Tuple of (reduced_features, reduced_feature_names, pca_reducer)
    """
    if pca_config is None:
        pca_config = PCAConfig()
    
    reducer = PCAFeatureReducer(pca_config)
    reduced_features, reduced_names = reducer.fit_transform(feature_matrix, feature_names)
    return reduced_features, reduced_names, reducer


def create_pca_config_for_time_domain(n_components: Optional[int] = None,
                                     variance_threshold: float = 0.95,
                                     keep_original: bool = False) -> PCAConfig:
    """
    Create PCA configuration specifically for time-domain features.
    
    Args:
        n_components: Number of PCA components (None for auto-selection)
        variance_threshold: Variance threshold for auto-selection
        keep_original: Whether to keep original features alongside PCA
        
    Returns:
        PCAConfig configured for time-domain features
    """
    return PCAConfig(
        n_components=n_components,
        variance_threshold=variance_threshold,
        apply_to_time_domain=True,
        apply_to_statistical_moments=True,
        apply_to_shape_factors=True,
        apply_to_frequency_domain=False,
        apply_to_spectral_features=False,
        apply_to_hht_features=False,
        standardize_features=True,
        keep_original_features=keep_original,
        pca_prefix="time_pca"
    )