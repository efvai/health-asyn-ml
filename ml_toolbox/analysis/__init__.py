"""
Analysis module for motor health ML pipeline

This module provides comprehensive analysis tools for:
- Feature extraction and importance analysis
- Cross-validation and model evaluation
- Multi-frequency comparison and analysis
"""

from .model_evaluation import (evaluate_model_cv, cv_shap)
from .class_separability import (
    compute_cohens_d,
    cohens_d_summary,
    build_separability_report,
    plot_separability,
)

__all__ = [
    'evaluate_model_cv',
    'cv_shap',

    # Class separability
    'compute_cohens_d',
    'cohens_d_summary',
    'build_separability_report',
    'plot_separability'
]

__version__ = "1.0.0"