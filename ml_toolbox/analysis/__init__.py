"""
Analysis module for motor health ML pipeline

This module provides comprehensive analysis tools for:
- Feature extraction and importance analysis
- Cross-validation and model evaluation
- Multi-frequency comparison and analysis
"""

from .model_evaluation import (evaluate_model_cv, cv_shap)

__all__ = [
    'evaluate_model_cv',
    'cv_shap',
]

__version__ = "1.0.0"