"""
Analysis module for motor health ML pipeline

This module provides comprehensive analysis tools for:
- Feature extraction and importance analysis
- Cross-validation and model evaluation
- Multi-frequency comparison and analysis
"""

# Import feature analysis utilities
from .feature_analysis import (
    extract_features_for_frequency,
    get_feature_importance_cv,
    analyze_feature_importance,
    plot_feature_importance_comparison,
    compare_top_features_across_frequencies
)

# Import cross-validation analysis utilities
from .cv_analysis import (
    evaluate_model_cv,
    plot_cv_scores_by_fold,
    plot_cv_results_comparison,
    create_performance_summary,
    run_comprehensive_frequency_analysis
)

__all__ = [
    # Feature analysis
    'extract_features_for_frequency',
    'get_feature_importance_cv',
    'analyze_feature_importance',
    'plot_feature_importance_comparison',
    'compare_top_features_across_frequencies',
    
    # CV analysis
    'evaluate_model_cv',
    'plot_cv_scores_by_fold',
    'plot_cv_results_comparison',
    'create_performance_summary',
    'run_comprehensive_frequency_analysis'
]

__version__ = "1.0.0"