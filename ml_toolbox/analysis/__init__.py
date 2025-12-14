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
    plot_permuted_importance_comparison,
    plot_mdi_importance_comparison,
    compare_top_features_across_frequencies
)

# Import cross-validation analysis utilities
from .cv_analysis import (
    plot_cv_scores_by_fold,
    plot_cv_results_comparison,
    create_performance_summary,
    run_comprehensive_frequency_analysis,
    evaluate_incremental_features_cv,
    plot_incremental_feature_performance,
    write_incremental_results_to_excel
)

from .model_evaluation import (evaluate_model_cv, cv_shap)
from .class_separability import (
    compute_cohens_d,
    cohens_d_summary,
    build_separability_report,
    plot_separability,
)

__all__ = [
    # Feature analysis
    'extract_features_for_frequency',
    'get_feature_importance_cv',
    'analyze_feature_importance',
    'plot_permuted_importance_comparison',
    'plot_mdi_importance_comparison',
    'compare_top_features_across_frequencies',
    
    # CV analysis
    'plot_cv_scores_by_fold',
    'plot_cv_results_comparison',
    'create_performance_summary',
    'run_comprehensive_frequency_analysis',
    'evaluate_incremental_features_cv',
    'plot_incremental_feature_performance',
    'write_incremental_results_to_excel',

    'evaluate_model_cv',
    'cv_shap',

    # Class separability
    'compute_cohens_d',
    'cohens_d_summary',
    'build_separability_report',
    'plot_separability'
]

__version__ = "1.0.0"