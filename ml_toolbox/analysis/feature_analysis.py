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
from pathlib import Path


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


def _calculate_permutation_feature_importance(model: RandomForestClassifier, X_val: np.ndarray, y_val: np.ndarray) -> np.ndarray:
    """
    Calculate permutation-based feature importance using sklearn's implementation
    
    Args:
        model: Trained RandomForestClassifier
        X_val: Validation feature matrix (left-out test set)
        y_val: Validation labels
        
    Returns:
        Permutation-based feature importances
    """
    from sklearn.inspection import permutation_importance
    
    # Use sklearn's permutation importance on validation set
    result = permutation_importance(
        model, X_val, y_val, 
        n_repeats=50,
        random_state=42,
        scoring='accuracy',
        n_jobs=-1 
    )
    
    # Extract the mean importances from the result
    return np.array(result.importances_mean)  # type: ignore


def get_feature_importance_cv(X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract permutation and MDI feature importance across CV folds using validation sets
    
    Args:
        X: Feature matrix
        y: Labels
        cv_folds: Number of CV folds (set to 1 for simple train/test split)
        
    Returns:
        Tuple of (permutation_importances, mdi_importances) across folds (computed on validation sets)
    """
    # Handle case where CV is disabled (cv_folds = 1) - use simple train/test split
    if cv_folds == 1:
        from sklearn.model_selection import train_test_split
        
        # Do a single train/test split (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model on train set
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        model.fit(X_train_scaled, y_train)
        
        # Calculate both permutation and MDI importance
        perm_importance = _calculate_permutation_feature_importance(model, X_test_scaled, y_test)
        mdi_importance = model.feature_importances_
        
        return np.array([perm_importance]), np.array([mdi_importance])  # Return as 2D arrays for consistency
    
    # Regular CV case
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    perm_importances = []
    mdi_importances = []
    
    for train_idx, val_idx in cv.split(X, y):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
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
        
        # Calculate both permutation and MDI importance on validation set
        mdi_importance = model.feature_importances_
        perm_importance = _calculate_permutation_feature_importance(model, X_val_scaled, y_val_fold)
        perm_importances.append(perm_importance)
        mdi_importances.append(mdi_importance)
    
    return np.array(perm_importances), np.array(mdi_importances)


def analyze_feature_importance(features: np.ndarray, labels: np.ndarray, 
                             feature_names: List[str], frequency: str,
                             cv_folds: int = 5) -> pd.DataFrame:
    """
    Comprehensive feature importance analysis including both permutation and MDI importance
    
    Args:
        features: Feature matrix
        labels: Labels
        feature_names: Feature names
        frequency: Frequency label for reporting
        cv_folds: Number of CV folds
        
    Returns:
        DataFrame with feature importance statistics (permutation and MDI)
    """
    print(f"Computing feature importance for {frequency}...")
    
    # Get CV feature importance (both permutation and MDI)
    cv_perm_importances, cv_mdi_importances = get_feature_importance_cv(features, labels, cv_folds)
    
    # Calculate permutation importance statistics
    mean_perm_importance = np.mean(cv_perm_importances, axis=0)
    std_perm_importance = np.std(cv_perm_importances, axis=0)
    
    # Calculate MDI importance statistics
    mean_mdi_importance = np.mean(cv_mdi_importances, axis=0)
    std_mdi_importance = np.std(cv_mdi_importances, axis=0)
    
    # Create results DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Importance': mean_perm_importance,  # Keep permutation as main importance for backward compatibility
        'Std_Importance': std_perm_importance,
        'Stability_Score': mean_perm_importance / (std_perm_importance + 1e-8),
        'Mean_MDI_Importance': mean_mdi_importance,
        'Std_MDI_Importance': std_mdi_importance,
        'MDI_Stability_Score': mean_mdi_importance / (std_mdi_importance + 1e-8),
        'Frequency': frequency
    }).sort_values('Mean_Importance', ascending=False)
    
    return importance_df


def write_feature_importance_to_excel(importance_results: Dict[str, pd.DataFrame], 
                                    output_path: str) -> None:
    """
    Write feature importance results to Excel format.
    
    Args:
        importance_results: Dictionary mapping frequency to importance DataFrame
        output_path: Path to save the Excel file (.xlsx)
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        
        # Sheet 1: Top Features Comparison (Top 20 features from each frequency)
        comparison_data = []
        for freq, importance_df in importance_results.items():
            top_features = importance_df.head(20)
            for rank, (_, row) in enumerate(top_features.iterrows(), 1):
                comparison_row = {
                    'Frequency': freq,
                    'Rank': rank,
                    'Feature': row['Feature'],
                    'Mean_Importance': row['Mean_Importance'],
                    'Std_Importance': row['Std_Importance'],
                    'Stability_Score': row['Stability_Score']
                }
                # Add MDI importance columns if they exist
                if 'Mean_MDI_Importance' in row:
                    comparison_row.update({
                        'Mean_MDI_Importance': row['Mean_MDI_Importance'],
                        'Std_MDI_Importance': row['Std_MDI_Importance'],
                        'MDI_Stability_Score': row['MDI_Stability_Score']
                    })
                comparison_data.append(comparison_row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_excel(writer, sheet_name='Top_Features_Comparison', index=False)
        
        # Sheet 2: Feature Ranking Matrix (Pivot table showing rank of each feature across frequencies)
        pivot_data = []
        all_features = set()
        for importance_df in importance_results.values():
            all_features.update(importance_df['Feature'].tolist())
        
        for feature in all_features:
            feature_row = {'Feature': feature}
            for freq, importance_df in importance_results.items():
                feature_data = importance_df[importance_df['Feature'] == feature]
                if not feature_data.empty:
                    rank = feature_data.index[0] + 1  # Convert to 1-based ranking
                    importance = feature_data['Mean_Importance'].iloc[0]
                    feature_row[f'{freq}_Rank'] = rank
                    feature_row[f'{freq}_Importance'] = importance
                else:
                    feature_row[f'{freq}_Rank'] = None
                    feature_row[f'{freq}_Importance'] = 0.0
            pivot_data.append(feature_row)
        
        ranking_df = pd.DataFrame(pivot_data)
        ranking_df.to_excel(writer, sheet_name='Feature_Ranking_Matrix', index=False)
        
        # Sheet 3: Detailed Results by Frequency
        detailed_data = []
        for freq, importance_df in importance_results.items():
            for _, row in importance_df.iterrows():
                detailed_row = {
                    'Frequency': freq,
                    'Feature': row['Feature'],
                    'Mean_Importance': row['Mean_Importance'],
                    'Std_Importance': row['Std_Importance'],
                    'Stability_Score': row['Stability_Score'],
                    'Feature_Type': row['Feature'].split('_')[0] if '_' in row['Feature'] else 'Unknown'
                }
                # Add MDI importance columns if they exist
                if 'Mean_MDI_Importance' in row:
                    detailed_row.update({
                        'Mean_MDI_Importance': row['Mean_MDI_Importance'],
                        'Std_MDI_Importance': row['Std_MDI_Importance'],
                        'MDI_Stability_Score': row['MDI_Stability_Score']
                    })
                detailed_data.append(detailed_row)
        
        detailed_df = pd.DataFrame(detailed_data)
        detailed_df.to_excel(writer, sheet_name='Detailed_Importance', index=False)
        
        # Sheet 4: Feature Type Analysis
        type_analysis = []
        for freq, importance_df in importance_results.items():
            # Group by feature type (first part before underscore)
            importance_df_copy = importance_df.copy()
            importance_df_copy['Feature_Type'] = importance_df_copy['Feature'].apply(
                lambda x: x.split('_')[0] if '_' in x else 'Unknown'
            )
            
            for feature_type in importance_df_copy['Feature_Type'].unique():
                type_features = importance_df_copy[importance_df_copy['Feature_Type'] == feature_type]
                type_analysis.append({
                    'Frequency': freq,
                    'Feature_Type': feature_type,
                    'Count': len(type_features),
                    'Mean_Importance': type_features['Mean_Importance'].mean(),
                    'Max_Importance': type_features['Mean_Importance'].max(),
                    'Top_Feature': type_features.iloc[0]['Feature'],
                    'Avg_Stability': type_features['Stability_Score'].mean()
                })
        
        type_analysis_df = pd.DataFrame(type_analysis)
        type_analysis_df.to_excel(writer, sheet_name='Feature_Type_Analysis', index=False)
        
        # Sheet 5: Analysis Info
        analysis_info = {
            'Property': [
                'Number of Frequencies Analyzed',
                'Total Unique Features',
                'Best Performing Frequency (by top feature)',
                'Most Consistent Frequency (by avg stability)',
                'Feature with Highest Importance',
                'Most Stable Feature (across all frequencies)',
                'Analysis Date'
            ],
            'Value': [
                len(importance_results),
                len(all_features),
                max(importance_results.keys(), 
                    key=lambda x: importance_results[x].iloc[0]['Mean_Importance']),
                max(importance_results.keys(), 
                    key=lambda x: importance_results[x]['Stability_Score'].mean()),
                max([(freq, df.iloc[0]['Feature'], df.iloc[0]['Mean_Importance']) 
                     for freq, df in importance_results.items()], 
                    key=lambda x: x[2])[1],
                max([(freq, df.iloc[0]['Feature'], df.iloc[0]['Stability_Score']) 
                     for freq, df in importance_results.items()], 
                    key=lambda x: x[2])[1],
                pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            ]
        }
        pd.DataFrame(analysis_info).to_excel(writer, sheet_name='Analysis_Info', index=False)
        
        # Sheet 7: Individual Frequency Sheets (detailed breakdown for each frequency)
        for freq, importance_df in importance_results.items():
            sheet_name = f'Details_{freq.upper()}'
            if len(sheet_name) > 31:  # Excel sheet name limit
                sheet_name = f'Det_{freq.upper()}'
            importance_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"Feature importance results written to Excel: {output_path}")


def plot_permuted_importance_comparison(importance_results: Dict[str, pd.DataFrame], 
                                     top_n: int = 15):
    """
    Plot permutation feature importance comparison across frequencies
    
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
        axes[i].set_xlabel('Permutation Feature Importance')
        axes[i].set_title(f'Top {top_n} Permutation Features: {freq.upper()}')
        axes[i].invert_yaxis()
    
    plt.tight_layout()
    plt.show()


def plot_mdi_importance_comparison(importance_results: Dict[str, pd.DataFrame], 
                                 top_n: int = 15):
    """
    Plot MDI feature importance comparison across frequencies
    
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
        
        # Check if MDI columns are available
        if 'Mean_MDI_Importance' not in df.columns:
            print(f"Warning: MDI importance not available for {freq}")
            continue
            
        # Sort by MDI importance and get top features
        mdi_sorted = df.sort_values('Mean_MDI_Importance', ascending=False)
        top_features = mdi_sorted.head(top_n)
        
        # Horizontal bar plot
        axes[i].barh(range(len(top_features)), top_features['Mean_MDI_Importance'], 
                    xerr=top_features['Std_MDI_Importance'], alpha=0.7, capsize=3, color='orange')
        axes[i].set_yticks(range(len(top_features)))
        axes[i].set_yticklabels([name.split('_')[-1][:15] for name in top_features['Feature']])
        axes[i].set_xlabel('MDI Feature Importance')
        axes[i].set_title(f'Top {top_n} MDI Features: {freq.upper()}')
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