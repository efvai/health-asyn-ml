"""
SHAP-based feature importance analysis for motor health monitoring
Provides explainable AI insights using SHAP values to complement OOB importance
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, List, Dict, Any, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import shap

def compute_shap_importance_per_class_cv(features: np.ndarray, labels: np.ndarray, 
                                        feature_names: List[str], cv_folds: int = 5) -> Dict[str, Any]:
    """
    Compute SHAP feature importance per class across CV folds
    
    Args:
        features: Feature matrix
        labels: Labels
        feature_names: Feature names
        cv_folds: Number of CV folds (set to 1 for simple train/test split)
        
    Returns:
        Dictionary with per-class SHAP results
    """   
    # Handle case where CV is disabled (cv_folds = 1) - use simple train/test split
    if cv_folds == 1:
        from sklearn.model_selection import train_test_split
        
        # Do a single train/test split (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Get unique classes
        unique_classes = np.unique(labels)
        n_classes = len(unique_classes)
        
        print(f"Computing per-class SHAP values using train/test split for {n_classes} classes...")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        model.fit(X_train_scaled, y_train)
        
        # Compute SHAP values on test set
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_scaled)
        
        # Handle different SHAP output formats and store per-class importance
        fold_class_importance = {}
        
        if isinstance(shap_values, list):  # Multi-class case (some versions)
            for class_idx, class_shap in enumerate(shap_values):
                class_label = unique_classes[class_idx]
                # Take absolute values and mean across samples for this class
                class_importance = np.mean(np.abs(class_shap), axis=0)
                fold_class_importance[class_label] = class_importance
                
        elif shap_values.ndim == 3:  # Multi-class case (newer versions): (samples, features, classes)
            shap_abs = np.abs(shap_values)
            for class_idx in range(n_classes):
                class_label = unique_classes[class_idx]
                # Extract SHAP values for this class and average across samples
                class_importance = np.mean(shap_abs[:, :, class_idx], axis=0)
                fold_class_importance[class_label] = class_importance
                
        else:  # Binary case: (samples, features)
            # For binary case, we have positive and negative contributions
            shap_abs_mean = np.abs(shap_values)
            class_importance = np.mean(shap_abs_mean, axis=0)
            fold_class_importance[unique_classes[0]] = class_importance
        
        # Format results for consistency with CV version
        per_class_results = {}
        for class_label in unique_classes:
            if class_label in fold_class_importance:
                per_class_results[class_label] = {
                    'mean_shap_importance': fold_class_importance[class_label],
                    'std_shap_importance': np.zeros_like(fold_class_importance[class_label]),  # No std for single split
                    'all_fold_importances': np.array([fold_class_importance[class_label]])
                }
        
        results = {
            'per_class_results': per_class_results,
            'feature_names': feature_names,
            'unique_classes': unique_classes,
            'cv_folds': 1,
            'models': [model],
            'scalers': [scaler],
            'fold_indices': [np.arange(len(X_test))]
        }
        
        return results
    
    # Regular CV case
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    class_shap_values_list = []
    models = []
    scalers = []
    fold_indices = []
    
    # Get unique classes
    unique_classes = np.unique(labels)
    n_classes = len(unique_classes)
    
    print(f"Computing per-class SHAP values across {cv_folds} folds for {n_classes} classes...")
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(features, labels)):
        print(f"Processing fold {fold + 1}/{cv_folds}")
        
        X_train_fold, X_val_fold = features[train_idx], features[val_idx]
        y_train_fold, y_val_fold = labels[train_idx], labels[val_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        model.fit(X_train_scaled, y_train_fold)
        

        X_val_shap = X_val_scaled
        y_val_shap = y_val_fold
        val_indices = val_idx
        
        # Compute SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val_shap)
        
        # Handle different SHAP output formats and store per-class importance
        fold_class_importance = {}
        
        if isinstance(shap_values, list):  # Multi-class case (some versions)
            for class_idx, class_shap in enumerate(shap_values):
                class_label = unique_classes[class_idx]
                # Take absolute values and mean across samples for this class
                class_importance = np.mean(np.abs(class_shap), axis=0)
                fold_class_importance[class_label] = class_importance
                
        elif shap_values.ndim == 3:  # Multi-class case (newer versions): (samples, features, classes)
            shap_abs = np.abs(shap_values)
            for class_idx in range(n_classes):
                class_label = unique_classes[class_idx]
                # Extract SHAP values for this class and average across samples
                class_importance = np.mean(shap_abs[:, :, class_idx], axis=0)
                fold_class_importance[class_label] = class_importance
                
        else:  # Binary case: (samples, features)
            # For binary case, we have positive and negative contributions
            shap_abs_mean = np.abs(shap_values)
            class_importance = np.mean(shap_abs_mean, axis=0)
            fold_class_importance[unique_classes[0]] = class_importance
        
        class_shap_values_list.append(fold_class_importance)
        models.append(model)
        scalers.append(scaler)
        fold_indices.append(val_indices)
    
    # Aggregate SHAP values across folds for each class
    per_class_results = {}
    
    for class_label in unique_classes:
        # Collect importance values for this class across all folds
        class_importances = []
        for fold_results in class_shap_values_list:
            if class_label in fold_results:
                class_importances.append(fold_results[class_label])
        
        if class_importances:
            class_importances = np.array(class_importances)
            mean_importance = np.mean(class_importances, axis=0)
            std_importance = np.std(class_importances, axis=0)
            
            per_class_results[class_label] = {
                'mean_shap_importance': mean_importance,
                'std_shap_importance': std_importance,
                'all_fold_importances': class_importances
            }
    
    results = {
        'per_class_results': per_class_results,
        'feature_names': feature_names,
        'unique_classes': unique_classes,
        'cv_folds': cv_folds,
        'models': models,
        'scalers': scalers,
        'fold_indices': fold_indices
    }
    
    return results


def analyze_shap_importance_per_class(features: np.ndarray, labels: np.ndarray, 
                                    feature_names: List[str], frequency: str,
                                    cv_folds: int = 5) -> Dict[str, pd.DataFrame]:
    """
    Comprehensive per-class SHAP feature importance analysis
    
    Args:
        features: Feature matrix
        labels: Labels
        feature_names: Feature names
        frequency: Frequency label for reporting
        cv_folds: Number of CV folds
        max_samples: Maximum samples for SHAP analysis
        
    Returns:
        Dictionary mapping class labels to importance DataFrames
    """
    print(f"Computing per-class SHAP importance for {frequency}...")
    
    # Get per-class SHAP importance
    shap_results = compute_shap_importance_per_class_cv(features, labels, feature_names, 
                                                      cv_folds)
    
    # Create results DataFrames for each class
    class_importance_dfs = {}
    
    for class_label, class_results in shap_results['per_class_results'].items():
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Mean_SHAP_Importance': class_results['mean_shap_importance'],
            'Std_SHAP_Importance': class_results['std_shap_importance'],
            'SHAP_Stability_Score': class_results['mean_shap_importance'] / (class_results['std_shap_importance'] + 1e-8),
            'Class': class_label,
            'Frequency': frequency
        }).sort_values('Mean_SHAP_Importance', ascending=False)
        
        class_importance_dfs[class_label] = importance_df
    
    return class_importance_dfs


def compute_shap_importance_cv(features: np.ndarray, labels: np.ndarray, 
                             feature_names: List[str], cv_folds: int = 5) -> Dict[str, Any]:
    """
    Compute SHAP feature importance across CV folds
    
    Args:
        features: Feature matrix
        labels: Labels
        feature_names: Feature names
        cv_folds: Number of CV folds (set to 1 for simple train/test split)
        
    Returns:
        Dictionary with SHAP results
    """   
    # Handle case where CV is disabled (cv_folds = 1) - use simple train/test split
    if cv_folds == 1:
        from sklearn.model_selection import train_test_split
        
        # Do a single train/test split (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print(f"Computing SHAP values using train/test split...")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        model.fit(X_train_scaled, y_train)
        
        # Compute SHAP values on test set
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_scaled)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):  # Multi-class case (some versions)
            # Take mean across classes first, then across samples
            shap_abs_mean = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            fold_importance = np.mean(shap_abs_mean, axis=0)
        elif shap_values.ndim == 3:  # Multi-class case (newer versions): (samples, features, classes)
            # Take absolute values, mean across classes, then mean across samples
            shap_abs = np.abs(shap_values)
            shap_mean_classes = np.mean(shap_abs, axis=2)  # Average across classes
            fold_importance = np.mean(shap_mean_classes, axis=0)  # Average across samples
        else:  # Binary case: (samples, features)
            shap_abs_mean = np.abs(shap_values)
            fold_importance = np.mean(shap_abs_mean, axis=0)
        
        # Format results for consistency with CV version
        results = {
            'feature_names': feature_names,
            'mean_shap_importance': fold_importance,
            'std_shap_importance': np.zeros_like(fold_importance),  # No std for single split
            'all_shap_importances': np.array([fold_importance]),
            'cv_folds': 1,
            'models': [model],
            'scalers': [scaler],
            'fold_indices': [np.arange(len(X_test))]
        }
        
        return results
    
    # Regular CV case
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    shap_values_list = []
    models = []
    scalers = []
    fold_indices = []
    
    print(f"Computing SHAP values across {cv_folds} folds...")
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(features, labels)):
        print(f"Processing fold {fold + 1}/{cv_folds}")
        
        X_train_fold, X_val_fold = features[train_idx], features[val_idx]
        y_train_fold, y_val_fold = labels[train_idx], labels[val_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        model.fit(X_train_scaled, y_train_fold)
        
        X_val_shap = X_val_scaled
        val_indices = val_idx
        
        # Compute SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val_shap)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):  # Multi-class case (some versions)
            # Take mean across classes first, then across samples
            shap_abs_mean = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            fold_importance = np.mean(shap_abs_mean, axis=0)
        elif shap_values.ndim == 3:  # Multi-class case (newer versions): (samples, features, classes)
            # Take absolute values, mean across classes, then mean across samples
            shap_abs = np.abs(shap_values)
            shap_mean_classes = np.mean(shap_abs, axis=2)  # Average across classes
            fold_importance = np.mean(shap_mean_classes, axis=0)  # Average across samples
        else:  # Binary case: (samples, features)
            shap_abs_mean = np.abs(shap_values)
            fold_importance = np.mean(shap_abs_mean, axis=0)
        
        # Store the mean importance per feature for this fold
        shap_values_list.append(fold_importance)
        models.append(model)
        scalers.append(scaler)
        fold_indices.append(val_indices)
    
    # Aggregate SHAP values across folds
    all_shap_importances = np.array(shap_values_list)
    
    # Calculate feature importance statistics
    mean_shap_importance = np.mean(all_shap_importances, axis=0)
    std_shap_importance = np.std(all_shap_importances, axis=0)
    
    results = {
        'feature_names': feature_names,
        'mean_shap_importance': mean_shap_importance,
        'std_shap_importance': std_shap_importance,
        'all_shap_importances': all_shap_importances,
        'cv_folds': cv_folds,
        'models': models,
        'scalers': scalers,
        'fold_indices': fold_indices
    }
    
    return results


def analyze_shap_importance(features: np.ndarray, labels: np.ndarray, 
                          feature_names: List[str], frequency: str,
                          cv_folds: int = 5) -> pd.DataFrame:
    """
    Comprehensive SHAP feature importance analysis
    
    Args:
        features: Feature matrix
        labels: Labels
        feature_names: Feature names
        frequency: Frequency label for reporting
        cv_folds: Number of CV folds
        max_samples: Maximum samples for SHAP analysis
        
    Returns:
        DataFrame with SHAP importance statistics
    """
    print(f"Computing SHAP importance for {frequency}...")
    
    # Get SHAP importance
    shap_results = compute_shap_importance_cv(features, labels, feature_names, 
                                            cv_folds)
    
    # Create results DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_SHAP_Importance': shap_results['mean_shap_importance'],
        'Std_SHAP_Importance': shap_results['std_shap_importance'],
        'SHAP_Stability_Score': shap_results['mean_shap_importance'] / (shap_results['std_shap_importance'] + 1e-8),
        'Frequency': frequency
    }).sort_values('Mean_SHAP_Importance', ascending=False)
    
    return importance_df


def compare_oob_vs_shap_importance(oob_importance_df: pd.DataFrame, 
                                 shap_importance_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare OOB (Random Forest) importance with SHAP importance, including MDI if available
    
    Args:
        oob_importance_df: DataFrame with OOB importance results (may include MDI)
        shap_importance_df: DataFrame with SHAP importance results
        
    Returns:
        Comparison DataFrame
    """
    # Prepare base columns for merging
    oob_cols = ['Feature', 'Mean_Importance', 'Stability_Score']
    
    # Add MDI columns if they exist in the OOB dataframe
    if 'Mean_MDI_Importance' in oob_importance_df.columns:
        oob_cols.extend(['Mean_MDI_Importance', 'MDI_Stability_Score'])
    
    # Merge the two DataFrames on feature names
    comparison = pd.merge(
        oob_importance_df[oob_cols],
        shap_importance_df[['Feature', 'Mean_SHAP_Importance', 'SHAP_Stability_Score']],
        on='Feature',
        how='inner'
    )
    
    # Calculate correlation and rank differences for permutation importance
    comparison['OOB_Rank'] = comparison['Mean_Importance'].rank(ascending=False, method='min').astype(int)
    comparison['SHAP_Rank'] = comparison['Mean_SHAP_Importance'].rank(ascending=False, method='min').astype(int)
    comparison['Rank_Difference'] = abs(comparison['OOB_Rank'] - comparison['SHAP_Rank'])
    
    # Normalize importances for comparison
    comparison['OOB_Normalized'] = comparison['Mean_Importance'] / comparison['Mean_Importance'].max()
    comparison['SHAP_Normalized'] = comparison['Mean_SHAP_Importance'] / comparison['Mean_SHAP_Importance'].max()
    comparison['Importance_Difference'] = abs(comparison['OOB_Normalized'] - comparison['SHAP_Normalized'])
    
    # Add MDI vs SHAP comparison if MDI data is available
    if 'Mean_MDI_Importance' in comparison.columns:
        comparison['MDI_Rank'] = comparison['Mean_MDI_Importance'].rank(ascending=False, method='min').astype(int)
        comparison['MDI_SHAP_Rank_Difference'] = abs(comparison['MDI_Rank'] - comparison['SHAP_Rank'])
        comparison['MDI_Normalized'] = comparison['Mean_MDI_Importance'] / comparison['Mean_MDI_Importance'].max()
        comparison['MDI_SHAP_Importance_Difference'] = abs(comparison['MDI_Normalized'] - comparison['SHAP_Normalized'])
        comparison['MDI_SHAP_Agreement_Score'] = 1 / (1 + comparison['MDI_SHAP_Rank_Difference'] + comparison['MDI_SHAP_Importance_Difference'])
    
    # Calculate agreement metrics (using permutation importance as primary)
    comparison['Agreement_Score'] = 1 / (1 + comparison['Rank_Difference'] + comparison['Importance_Difference'])
    
    return comparison.sort_values('Agreement_Score', ascending=False)


def plot_shap_importance_per_class(class_importance_results: Dict[str, pd.DataFrame], 
                                  top_n: int = 15, frequency: str = ""):
    """
    Plot SHAP importance for each class separately
    
    Args:
        class_importance_results: Dict mapping class labels to importance DataFrames
        top_n: Number of top features to show per class
        frequency: Frequency label for title
    """
    n_classes = len(class_importance_results)
    
    # Calculate subplot layout
    cols = min(2, n_classes)
    rows = (n_classes + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if n_classes == 1:
        axes = [axes]
    elif rows == 1 and cols > 1:
        axes = axes
    elif rows > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold', 'lightpink', 'lightyellow']
    class_names = ['Healthy', 'Faulty Bearing', 'Misalignment', 'System Misalignment']
    
    for i, (class_label, df) in enumerate(class_importance_results.items()):
        if i >= len(axes):
            break
            
        top_features = df.head(top_n)
        
        # Horizontal bar plot
        y_pos = range(len(top_features))
        axes[i].barh(y_pos, top_features['Mean_SHAP_Importance'], 
                    xerr=top_features['Std_SHAP_Importance'], 
                    alpha=0.7, capsize=3, color=colors[i % len(colors)])
        
        axes[i].set_yticks(y_pos)
        axes[i].set_yticklabels([name.split('_')[-1][:15] for name in top_features['Feature']])
        axes[i].set_xlabel('SHAP Importance')
        
        # Use descriptive class names
        class_name = class_names[int(class_label)] if int(class_label) < len(class_names) else f'Class {class_label}'
        title = f'Top {top_n} Features: {class_name}'
        if frequency:
            title += f' ({frequency.upper()})'
        axes[i].set_title(title)
        axes[i].invert_yaxis()
        
        # Add value labels on bars
        for j, (_, row) in enumerate(top_features.iterrows()):
            axes[i].text(row['Mean_SHAP_Importance'] + 0.001, j, 
                        f'{row["Mean_SHAP_Importance"]:.3f}', 
                        ha='left', va='center', fontsize=8)
    
    # Hide unused subplots
    for i in range(n_classes, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_shap_class_comparison_heatmap(class_importance_results: Dict[str, pd.DataFrame], 
                                     top_n: int = 20, frequency: str = ""):
    """
    Plot heatmap comparing feature importance across classes
    
    Args:
        class_importance_results: Dict mapping class labels to importance DataFrames
        top_n: Number of top features to include
        frequency: Frequency label for title
    """
    # Get top features from each class
    all_top_features = set()
    for df in class_importance_results.values():
        all_top_features.update(df.head(top_n)['Feature'].tolist())
    
    # Create matrix of importance values
    feature_list = list(all_top_features)
    class_labels = list(class_importance_results.keys())
    class_names = ['Healthy', 'Faulty Bearing', 'Misalignment', 'System Misalignment']
    
    importance_matrix = np.zeros((len(feature_list), len(class_labels)))
    
    for j, class_label in enumerate(class_labels):
        df = class_importance_results[class_label]
        for i, feature in enumerate(feature_list):
            feature_row = df[df['Feature'] == feature]
            if not feature_row.empty:
                importance_matrix[i, j] = feature_row['Mean_SHAP_Importance'].iloc[0]
    
    # Create DataFrame for heatmap
    heatmap_df = pd.DataFrame(
        importance_matrix, 
        index=[f.split('_')[-1][:20] for f in feature_list],
        columns=[class_names[int(cl)] if int(cl) < len(class_names) else f'Class {cl}' 
                for cl in class_labels]
    )
    
    # Plot heatmap
    plt.figure(figsize=(12, max(8, len(feature_list) * 0.3)))
    sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='YlOrRd', 
                cbar_kws={'label': 'SHAP Importance'})
    
    title = 'SHAP Feature Importance by Class'
    if frequency:
        title += f' ({frequency.upper()})'
    plt.title(title)
    plt.ylabel('Features')
    plt.xlabel('Motor Health Condition')
    plt.tight_layout()
    plt.show()


def plot_shap_importance_comparison(shap_importance_results: Dict[str, pd.DataFrame], 
                                  top_n: int = 15):
    """
    Plot SHAP importance comparison across frequencies
    
    Args:
        shap_importance_results: Dict mapping frequency to SHAP importance DataFrame
        top_n: Number of top features to show
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    axes = axes.flatten()
    
    frequencies = list(shap_importance_results.keys())
    
    for i, freq in enumerate(frequencies):
        if i >= 4:  # Max 4 subplots
            break
            
        df = shap_importance_results[freq]
        top_features = df.head(top_n)
        
        # Horizontal bar plot
        axes[i].barh(range(len(top_features)), top_features['Mean_SHAP_Importance'], 
                    xerr=top_features['Std_SHAP_Importance'], alpha=0.7, capsize=3,
                    color='orange')
        axes[i].set_yticks(range(len(top_features)))
        axes[i].set_yticklabels(top_features['Feature'].tolist())
        axes[i].set_xlabel('SHAP Importance')
        axes[i].set_title(f'Top {top_n} Features (SHAP): {freq.upper()}')
        axes[i].invert_yaxis()
    
    plt.tight_layout()
    plt.show()


def plot_oob_vs_shap_comparison(comparison_df: pd.DataFrame, top_n: int = 20):
    """
    Plot comparison between OOB and SHAP importance
    
    Args:
        comparison_df: Comparison DataFrame from compare_oob_vs_shap_importance
        top_n: Number of top features to show
    """
    top_features = comparison_df.head(top_n)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Scatter plot of normalized importances
    axes[0, 0].scatter(top_features['OOB_Normalized'], top_features['SHAP_Normalized'], 
                      alpha=0.7, s=60)
    axes[0, 0].plot([0, 1], [0, 1], 'r--', alpha=0.5)
    axes[0, 0].set_xlabel('OOB Importance (Normalized)')
    axes[0, 0].set_ylabel('SHAP Importance (Normalized)')
    axes[0, 0].set_title('OOB vs SHAP Importance')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Rank comparison
    axes[0, 1].scatter(top_features['OOB_Rank'], top_features['SHAP_Rank'], 
                      alpha=0.7, s=60, color='green')
    axes[0, 1].plot([1, top_n], [1, top_n], 'r--', alpha=0.5)
    axes[0, 1].set_xlabel('OOB Rank')
    axes[0, 1].set_ylabel('SHAP Rank')
    axes[0, 1].set_title('Ranking Comparison')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Agreement scores
    axes[1, 0].bar(range(len(top_features)), top_features['Agreement_Score'], 
                  alpha=0.7, color='purple')
    axes[1, 0].set_xlabel('Feature Index')
    axes[1, 0].set_ylabel('Agreement Score')
    axes[1, 0].set_title('Feature Agreement Scores')
    axes[1, 0].set_xticks(range(0, len(top_features), 5))
    
    # 4. Rank differences
    axes[1, 1].bar(range(len(top_features)), top_features['Rank_Difference'], 
                  alpha=0.7, color='red')
    axes[1, 1].set_xlabel('Feature Index')
    axes[1, 1].set_ylabel('Rank Difference')
    axes[1, 1].set_title('Rank Differences (|OOB - SHAP|)')
    axes[1, 1].set_xticks(range(0, len(top_features), 5))
    
    plt.tight_layout()
    plt.show()
    
    # Print correlation statistics
    correlation = np.corrcoef(comparison_df['OOB_Normalized'], comparison_df['SHAP_Normalized'])[0, 1]
    rank_correlation = np.corrcoef(comparison_df['OOB_Rank'], comparison_df['SHAP_Rank'])[0, 1]
    
    print(f"\nCORRELATION ANALYSIS:")
    print(f"Importance correlation: {correlation:.3f}")
    print(f"Rank correlation: {rank_correlation:.3f}")
    print(f"Mean agreement score: {comparison_df['Agreement_Score'].mean():.3f}")
    print(f"Mean rank difference: {comparison_df['Rank_Difference'].mean():.1f}")


def write_shap_per_class_to_excel(class_importance_results: Dict[str, Dict[str, pd.DataFrame]], 
                                output_path: str, frequency: str = "") -> None:
    """
    Write per-class SHAP importance results to Excel format.
    
    Args:
        class_importance_results: Dictionary mapping frequency to dict of class importance DataFrames
        output_path: Path to save the Excel file (.xlsx)
        frequency: Frequency label for single frequency analysis
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    class_names = ['Healthy', 'Faulty Bearing', 'Misalignment', 'System Misalignment']
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        
        # If single frequency analysis
        if frequency and frequency in class_importance_results:
            class_dfs = class_importance_results[frequency]
            
            # Sheet 1: Summary by Class
            summary_data = []
            for class_label, df in class_dfs.items():
                top_feature = df.iloc[0]
                class_name = class_names[int(class_label)] if int(class_label) < len(class_names) else f'Class {class_label}'
                
                summary_data.append({
                    'Class_Label': class_label,
                    'Class_Name': class_name,
                    'Top_Feature': top_feature['Feature'],
                    'Top_Feature_Importance': top_feature['Mean_SHAP_Importance'],
                    'Top_Feature_Stability': top_feature['SHAP_Stability_Score'],
                    'Mean_Importance_All': df['Mean_SHAP_Importance'].mean(),
                    'Features_Above_001': len(df[df['Mean_SHAP_Importance'] > 0.01]),
                    'Features_Above_005': len(df[df['Mean_SHAP_Importance'] > 0.05]),
                    'N_Features': len(df)
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Class_Summary', index=False)
            
            # Individual class sheets
            for class_label, df in class_dfs.items():
                class_name = class_names[int(class_label)] if int(class_label) < len(class_names) else f'Class_{class_label}'
                sheet_name = f'{class_name}_Details'[:31]  # Excel sheet name limit
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Sheet: Top Features Comparison
            comparison_data = []
            for class_label, df in class_dfs.items():
                class_name = class_names[int(class_label)] if int(class_label) < len(class_names) else f'Class {class_label}'
                top_features = df.head(15)
                for rank, (_, row) in enumerate(top_features.iterrows(), 1):
                    comparison_data.append({
                        'Class_Label': class_label,
                        'Class_Name': class_name,
                        'Rank': rank,
                        'Feature': row['Feature'],
                        'Mean_SHAP_Importance': row['Mean_SHAP_Importance'],
                        'Std_SHAP_Importance': row['Std_SHAP_Importance'],
                        'SHAP_Stability_Score': row['SHAP_Stability_Score']
                    })
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df.to_excel(writer, sheet_name='Top_Features_by_Class', index=False)
            
            # Sheet: Feature Ranking Matrix
            all_features = set()
            for df in class_dfs.values():
                all_features.update(df['Feature'].tolist())
            
            ranking_data = []
            for feature in all_features:
                feature_row = {'Feature': feature}
                for class_label, df in class_dfs.items():
                    class_name = class_names[int(class_label)] if int(class_label) < len(class_names) else f'Class_{class_label}'
                    feature_data = df[df['Feature'] == feature]
                    if not feature_data.empty:
                        rank = feature_data.index[0] + 1
                        importance = feature_data['Mean_SHAP_Importance'].iloc[0]
                        feature_row[f'{class_name}_Rank'] = rank
                        feature_row[f'{class_name}_Importance'] = importance
                    else:
                        feature_row[f'{class_name}_Rank'] = None
                        feature_row[f'{class_name}_Importance'] = 0.0
                ranking_data.append(feature_row)
            
            ranking_df = pd.DataFrame(ranking_data)
            ranking_df.to_excel(writer, sheet_name='Feature_Ranking_Matrix', index=False)
            
        # Multiple frequencies analysis
        else:
            # Implementation for multiple frequencies would go here
            # For now, we'll handle single frequency case
            pass
        
        # Analysis info sheet
        analysis_info = {
            'Property': [
                'Analysis Type',
                'Frequency Analyzed',
                'Number of Classes',
                'Class Names',
                'Analysis Date'
            ],
            'Value': [
                'Per-Class SHAP Feature Importance',
                frequency if frequency else 'Multiple',
                len(class_dfs) if frequency else 'Multiple',
                ', '.join([class_names[int(cl)] if int(cl) < len(class_names) else f'Class {cl}' 
                          for cl in class_dfs.keys()]) if frequency else 'Multiple',
                pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            ]
        }
        pd.DataFrame(analysis_info).to_excel(writer, sheet_name='Analysis_Info', index=False)
    
    print(f"Per-class SHAP results written to Excel: {output_path}")
    if frequency:
        print(f"Classes analyzed: {list(class_dfs.keys())}")
        print(f"Sheets created: Class_Summary, individual class details, Top_Features_by_Class, Feature_Ranking_Matrix, Analysis_Info")


def write_shap_comparison_to_excel(oob_results: Dict[str, pd.DataFrame],
                                 shap_results: Dict[str, pd.DataFrame],
                                 output_path: str) -> None:
    """
    Write OOB vs SHAP comparison results to Excel format.
    
    Args:
        oob_results: Dictionary mapping frequency to OOB importance DataFrame
        shap_results: Dictionary mapping frequency to SHAP importance DataFrame
        output_path: Path to save the Excel file (.xlsx)
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        
        # Sheet 1: Summary Comparison
        summary_data = []
        for freq in oob_results.keys():
            if freq in shap_results:
                oob_df = oob_results[freq]
                shap_df = shap_results[freq]
                comparison = compare_oob_vs_shap_importance(oob_df, shap_df)
                
                correlation = np.corrcoef(comparison['OOB_Normalized'], 
                                        comparison['SHAP_Normalized'])[0, 1]
                rank_correlation = np.corrcoef(comparison['OOB_Rank'], 
                                             comparison['SHAP_Rank'])[0, 1]
                
                summary_entry = {
                    'Frequency': freq,
                    'Importance_Correlation': correlation,
                    'Rank_Correlation': rank_correlation,
                    'Mean_Agreement_Score': comparison['Agreement_Score'].mean(),
                    'Mean_Rank_Difference': comparison['Rank_Difference'].mean(),
                    'Top_OOB_Feature': oob_df.iloc[0]['Feature'],
                    'Top_SHAP_Feature': shap_df.iloc[0]['Feature'],
                    'Top_Features_Match': oob_df.iloc[0]['Feature'] == shap_df.iloc[0]['Feature']
                }
                
                # Add MDI comparison metrics if available
                if 'MDI_SHAP_Agreement_Score' in comparison.columns:
                    mdi_correlation = np.corrcoef(comparison['MDI_Normalized'], 
                                                comparison['SHAP_Normalized'])[0, 1]
                    mdi_rank_correlation = np.corrcoef(comparison['MDI_Rank'], 
                                                     comparison['SHAP_Rank'])[0, 1]
                    
                    # Find top MDI feature
                    top_mdi_feature = 'N/A'
                    if 'Mean_MDI_Importance' in oob_df.columns:
                        mdi_sorted = oob_df.sort_values('Mean_MDI_Importance', ascending=False)
                        top_mdi_feature = mdi_sorted.iloc[0]['Feature']
                    
                    summary_entry.update({
                        'MDI_SHAP_Importance_Correlation': mdi_correlation,
                        'MDI_SHAP_Rank_Correlation': mdi_rank_correlation,
                        'Mean_MDI_SHAP_Agreement_Score': comparison['MDI_SHAP_Agreement_Score'].mean(),
                        'Mean_MDI_SHAP_Rank_Difference': comparison['MDI_SHAP_Rank_Difference'].mean(),
                        'Top_MDI_Feature': top_mdi_feature
                    })
                
                summary_data.append(summary_entry)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Comparison_Summary', index=False)
        
        # Sheet 2: Detailed comparison for each frequency
        for freq in oob_results.keys():
            if freq in shap_results:
                comparison = compare_oob_vs_shap_importance(oob_results[freq], shap_results[freq])
                sheet_name = f'Comparison_{freq.upper()}'
                if len(sheet_name) > 31:
                    sheet_name = f'Comp_{freq.upper()}'
                comparison.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Sheet 3: SHAP Summary
        shap_summary_data = []
        for freq, shap_df in shap_results.items():
            top_feature = shap_df.iloc[0]
            shap_summary_data.append({
                'Frequency': freq,
                'Top_SHAP_Feature': top_feature['Feature'],
                'Top_SHAP_Importance': top_feature['Mean_SHAP_Importance'],
                'Top_SHAP_Stability': top_feature['SHAP_Stability_Score'],
                'Mean_SHAP_Importance': shap_df['Mean_SHAP_Importance'].mean(),
                'N_Features': len(shap_df)
            })
        
        shap_summary_df = pd.DataFrame(shap_summary_data)
        shap_summary_df.to_excel(writer, sheet_name='SHAP_Summary', index=False)
        
        # Sheet 4: Analysis Info
        analysis_info = {
            'Property': [
                'Analysis Type',
                'Frequencies Analyzed',
                'Best Correlation Frequency',
                'Most Consistent Frequency (SHAP)',
                'Overall Mean Correlation',
                'Analysis Date'
            ],
            'Value': [
                'OOB vs SHAP Comparison',
                list(oob_results.keys()),
                max(summary_data, key=lambda x: x['Importance_Correlation'])['Frequency'] if summary_data else 'N/A',
                max(shap_results.keys(), key=lambda x: shap_results[x]['SHAP_Stability_Score'].mean()) if shap_results else 'N/A',
                np.mean([x['Importance_Correlation'] for x in summary_data]) if summary_data else 'N/A',
                pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            ]
        }
        pd.DataFrame(analysis_info).to_excel(writer, sheet_name='Analysis_Info', index=False)
    
    print(f"OOB vs SHAP comparison written to Excel: {output_path}")
    print(f"Sheets created: Comparison_Summary, individual frequency comparisons, SHAP_Summary, Analysis_Info")


def run_comprehensive_shap_analysis(data_loader, frequencies: List[str], 
                                   load: str = "no load", 
                                   max_windows_per_class: int = 20,
                                   window_length: int = 1024,
                                   window_overlap: float = 0.5,
                                   export_to_excel: bool = False,
                                   output_dir: str = "output",
                                   include_per_class: bool = True,
                                   cv_folds: int = 5) -> Tuple[Dict, Dict, Dict]:
    """
    Run comprehensive SHAP analysis across multiple frequencies including per-class analysis
    
    Args:
        data_loader: DataLoader instance
        frequencies: List of frequencies to analyze
        load: Load condition
        max_windows_per_class: Maximum number of windows per class
        window_length: Window length for analysis
        window_overlap: Window overlap ratio
        export_to_excel: Whether to export results to Excel files
        output_dir: Directory to save Excel files
        include_per_class: Whether to include per-class SHAP analysis
        cv_folds: Number of CV folds (set to 1 for simple train/test split)
        
    Returns:
        Tuple of (oob_results, shap_results, per_class_shap_results)
    """
    from ml_toolbox.analysis.feature_analysis import (
        extract_features_for_frequency, 
        analyze_feature_importance
    )
       
    oob_results = {}
    shap_results = {}
    per_class_shap_results = {}
    
    print(f"Starting comprehensive SHAP analysis for frequencies: {frequencies}")
    print(f"Load condition: {load}")
    print(f"Max windows per class: {max_windows_per_class}")
    print(f"CV folds: {cv_folds}")
    print(f"Per-class analysis: {'Enabled' if include_per_class else 'Disabled'}")
    if export_to_excel:
        print(f"Excel export enabled - output directory: {output_dir}")
    print("=" * 60)
    
    for freq in frequencies:
        try:
            # Extract features
            features, labels, feature_names, metadata = extract_features_for_frequency(
                data_loader, freq, load, max_windows_per_class=max_windows_per_class, 
                window_size=window_length, overlap_ratio=window_overlap
            )
            
            # Analyze OOB importance
            oob_result = analyze_feature_importance(features, labels, feature_names, freq, cv_folds=cv_folds)
            oob_results[freq] = oob_result
            
            # Analyze overall SHAP importance
            shap_result = analyze_shap_importance(features, labels, feature_names, freq, cv_folds=cv_folds)
            shap_results[freq] = shap_result
            
            # Analyze per-class SHAP importance if requested
            if include_per_class:
                print(f"Computing per-class SHAP analysis for {freq}...")
                per_class_result = analyze_shap_importance_per_class(features, labels, feature_names, freq, cv_folds=cv_folds)
                per_class_shap_results[freq] = per_class_result
                
                # Export per-class results to Excel if requested
                if export_to_excel:
                    per_class_excel_path = f"{output_dir}/shap_per_class_{freq}_{load.replace(' ', '_')}.xlsx"
                    write_shap_per_class_to_excel({freq: per_class_result}, per_class_excel_path, frequency=freq)
            
            print(f"{freq} analysis completed")
            
        except Exception as e:
            print(f"Error analyzing {freq}: {str(e)}")
            continue
    
    # Export comparison to Excel if requested
    if export_to_excel and oob_results and shap_results:
        comparison_excel_path = f"{output_dir}/oob_vs_shap_comparison_{load.replace(' ', '_')}.xlsx"
        write_shap_comparison_to_excel(oob_results, shap_results, comparison_excel_path)
    
    print("=" * 60)
    print(f"SHAP analysis completed for {len(shap_results)} frequencies")
    if include_per_class:
        print(f"Per-class analysis completed for {len(per_class_shap_results)} frequencies")
    
    return oob_results, shap_results, per_class_shap_results