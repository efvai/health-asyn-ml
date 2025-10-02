"""
Test Excel export functionality for features and CV results.
This script demonstrates how to use the new Excel export functions.
"""

import numpy as np
import pandas as pd
from ml_toolbox.analysis.cv_analysis import write_features_to_excel, write_cv_results_to_excel

def test_excel_export():
    """Test Excel export functionality with sample data."""
    
    # Create sample features data
    n_samples = 100
    n_features = 50
    
    # Generate sample features
    np.random.seed(42)
    features = np.random.randn(n_samples, n_features)
    
    # Generate sample labels (4 classes: healthy, faulty_bearing, misalignment, system_misalignment)
    labels = np.random.choice(['healthy', 'faulty_bearing', 'misalignment', 'system_misalignment'], 
                             size=n_samples, p=[0.4, 0.2, 0.2, 0.2])
    
    # Generate feature names
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    
    # Generate sample metadata
    metadata = []
    for i in range(n_samples):
        metadata.append({
            'frequency': np.random.choice(['10hz', '20hz', '30hz', '40hz']),
            'load': np.random.choice(['no_load', 'under_load']),
            'sensor_type': 'current',
            'run_id': f'run_{i//10 + 1}'
        })
    
    print("Testing Excel export for features...")
    
    # Test features export
    features_output = "test_output/test_features.xlsx"
    write_features_to_excel(features, labels, feature_names, features_output, 
                           frequency="20hz", metadata=metadata)
    
    print("\nTesting Excel export for CV results...")
    
    # Create sample CV results
    cv_results = {}
    frequencies = ['10hz', '20hz', '30hz', '40hz']
    
    for freq in frequencies:
        cv_scores = np.random.uniform(0.7, 0.95, 5)  # 5-fold CV
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        cv_results[freq] = {
            'frequency': freq,
            'cv_scores': cv_scores,
            'mean_accuracy': cv_scores.mean(),
            'std_accuracy': cv_scores.std(),
            'best_fold': cv_scores.max(),
            'worst_fold': cv_scores.min(),
            'label_distribution': dict(zip(unique_labels, counts)),
            'n_samples': len(labels),
            'n_features': n_features
        }
    
    # Test CV results export
    cv_output = "test_output/test_cv_results.xlsx"
    write_cv_results_to_excel(cv_results, cv_output)
    
    print(f"\nExcel files created:")
    print(f"  - Features: {features_output}")
    print(f"  - CV Results: {cv_output}")
    print("\nYou can now open these files in Excel to inspect the data!")

if __name__ == "__main__":
    test_excel_export()