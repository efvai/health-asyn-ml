"""
Feature extraction and analysis utilities for motor health monitoring
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns


def extract_features_for_frequency(data_loader, frequency: str, load: str = "no load", 
                                 window_size: int = 1024, overlap_ratio: float = 0.5,
                                 max_windows_per_class: int = 20) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict]]:
    """
    Extract features for a specific frequency and load condition
    
    Args:
        data_loader: DataLoader instance
        frequency: Motor frequency (e.g., "10hz", "20hz", "30hz", "40hz")
        load: Load condition ("no load" or "under load")
        window_size: Size of windows for feature extraction
        overlap_ratio: Overlap ratio between windows
        max_windows_per_class: Maximum windows per class
        
    Returns:
        features: Feature matrix
        labels: Label array
        feature_names: List of feature names
        metadata: Window metadata
    """
    from ml_toolbox.data_loader import create_windows_for_ml, extract_features_for_ml
    
    print(f"Loading {frequency} {load} data...")
    
    # Load current sensor data
    current_data, current_metadata = data_loader.load_batch(
        sensor_type="current",
        frequency=frequency,
        load=load,
        max_workers=1
    )
    
    print(f"Loaded {len(current_data)} current sensor files for {frequency}")
    
    if not current_data:
        raise ValueError(f"No data found for {frequency} {load}")
    
    # Create windows
    windows, labels, win_metadata = create_windows_for_ml(
        current_data, current_metadata,
        window_size=window_size,
        overlap_ratio=overlap_ratio,
        max_windows_per_class=max_windows_per_class
    )
    
    print(f"Created {len(windows)} windows for {frequency}")
    
    # Extract features
    features, feature_names = extract_features_for_ml(
        windows,
        sensor_type="current",
        metadata_list=win_metadata
    )
    
    print(f"Extracted {features.shape[1]} features for {frequency}")
    
    return features, labels, feature_names, win_metadata


def get_feature_importance_cv(X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> np.ndarray:
    """
    Extract feature importance across CV folds
    
    Args:
        X: Feature matrix
        y: Labels
        cv_folds: Number of CV folds
        
    Returns:
        Feature importances across folds
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    feature_importances = []
    
    for train_idx, val_idx in cv.split(X, y):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        model.fit(X_train_scaled, y_train_fold)
        
        feature_importances.append(model.feature_importances_)
    
    return np.array(feature_importances)


def analyze_feature_importance(features: np.ndarray, labels: np.ndarray, 
                             feature_names: List[str], frequency: str,
                             cv_folds: int = 5) -> pd.DataFrame:
    """
    Comprehensive feature importance analysis
    
    Args:
        features: Feature matrix
        labels: Labels
        feature_names: Feature names
        frequency: Frequency label for reporting
        cv_folds: Number of CV folds
        
    Returns:
        DataFrame with feature importance statistics
    """
    print(f"Computing feature importance for {frequency}...")
    
    # Get CV feature importance
    cv_feature_importances = get_feature_importance_cv(features, labels, cv_folds)
    
    # Calculate statistics
    mean_importance = np.mean(cv_feature_importances, axis=0)
    std_importance = np.std(cv_feature_importances, axis=0)
    
    # Create results DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Importance': mean_importance,
        'Std_Importance': std_importance,
        'Stability_Score': mean_importance / (std_importance + 1e-8),
        'Frequency': frequency
    }).sort_values('Mean_Importance', ascending=False)
    
    return importance_df


def plot_feature_importance_comparison(importance_results: Dict[str, pd.DataFrame], 
                                     top_n: int = 15):
    """
    Plot feature importance comparison across frequencies
    
    Args:
        importance_results: Dict mapping frequency to importance DataFrame
        top_n: Number of top features to show
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    axes = axes.flatten()
    
    frequencies = list(importance_results.keys())
    
    for i, freq in enumerate(frequencies):
        if i >= 4:  # Max 4 subplots
            break
            
        df = importance_results[freq]
        top_features = df.head(top_n)
        
        # Horizontal bar plot
        axes[i].barh(range(len(top_features)), top_features['Mean_Importance'], 
                    xerr=top_features['Std_Importance'], alpha=0.7, capsize=3)
        axes[i].set_yticks(range(len(top_features)))
        axes[i].set_yticklabels([name.split('_')[-1][:15] for name in top_features['Feature']])
        axes[i].set_xlabel('Feature Importance')
        axes[i].set_title(f'Top {top_n} Features: {freq.upper()}')
        axes[i].invert_yaxis()
    
    plt.tight_layout()
    plt.show()


def compare_top_features_across_frequencies(importance_results: Dict[str, pd.DataFrame], 
                                          top_n: int = 10) -> pd.DataFrame:
    """
    Compare top features across different frequencies
    
    Args:
        importance_results: Dict mapping frequency to importance DataFrame
        top_n: Number of top features to compare
        
    Returns:
        Comparison DataFrame
    """
    comparison_data = []
    
    for freq, df in importance_results.items():
        top_features = df.head(top_n)
        for rank, (_, row) in enumerate(top_features.iterrows(), 1):
            comparison_data.append({
                'Frequency': freq,
                'Rank': rank,
                'Feature': row['Feature'],
                'Importance': row['Mean_Importance'],
                'Stability': row['Stability_Score']
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create pivot table for heatmap
    pivot_df = comparison_df.pivot(index='Feature', columns='Frequency', values='Rank')
    
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_df, annot=True, cmap='viridis_r', fmt='.0f', 
                cbar_kws={'label': 'Feature Rank'})
    plt.title('Feature Ranking Across Frequencies')
    plt.ylabel('Features')
    plt.xlabel('Frequency')
    plt.tight_layout()
    plt.show()
    
    return comparison_df