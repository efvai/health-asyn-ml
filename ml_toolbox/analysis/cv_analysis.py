"""
Cross-validation and model evaluation utilities
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path


def write_features_to_excel(features: np.ndarray, labels: np.ndarray, 
                           feature_names: List[str], output_path: str,
                           frequency: Optional[str] = None, 
                           metadata: Optional[List[Dict]] = None) -> None:
    """
    Write features matrix, labels, and feature names to Excel format for inspection.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        labels: Labels array (n_samples,)
        feature_names: List of feature names
        output_path: Path to save the Excel file (.xlsx)
        frequency: Optional frequency identifier
        metadata: Optional metadata list for each sample
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create Excel writer
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        
        # Sheet 1: Dataset Info
        dataset_info = {
            'Property': ['Number of Samples', 'Number of Features', 'Frequency', 'Created At'],
            'Value': [
                features.shape[0], 
                features.shape[1], 
                frequency if frequency else 'N/A',
                pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            ]
        }
        pd.DataFrame(dataset_info).to_excel(writer, sheet_name='Dataset_Info', index=False)
        
        # Sheet 2: Feature Names
        feature_names_df = pd.DataFrame({
            'Index': range(len(feature_names)),
            'Feature_Name': feature_names
        })
        feature_names_df.to_excel(writer, sheet_name='Feature_Names', index=False)
        
        # Sheet 3: Label Distribution
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        label_dist_df = pd.DataFrame({
            'Class': unique_labels,
            'Count': label_counts,
            'Percentage': (label_counts / len(labels) * 100).round(2)
        })
        label_dist_df.to_excel(writer, sheet_name='Label_Distribution', index=False)
        
        # Sheet 4: Features Matrix with Labels
        features_df = pd.DataFrame(features, columns=feature_names)
        features_df['Label'] = labels
        features_df['Sample_Index'] = range(len(labels))
        
        # Add metadata columns if available
        if metadata:
            for i, meta in enumerate(metadata):
                if i < len(features_df):
                    for key, value in meta.items():
                        col_name = f'Meta_{key}'
                        if col_name not in features_df.columns:
                            features_df[col_name] = None
                        features_df.loc[i, col_name] = value
        
        # Reorder columns: Sample_Index, Label, metadata, then features
        cols = ['Sample_Index', 'Label']
        meta_cols = [col for col in features_df.columns if col.startswith('Meta_')]
        feature_cols = [col for col in features_df.columns if col not in cols and not col.startswith('Meta_')]
        features_df = features_df[cols + meta_cols + feature_cols]
        
        features_df.to_excel(writer, sheet_name='Features_Data', index=False)
        
        # Sheet 5: Feature Statistics
        feature_stats = pd.DataFrame({
            'Feature_Name': feature_names,
            'Mean': np.mean(features, axis=0),
            'Std': np.std(features, axis=0),
            'Min': np.min(features, axis=0),
            'Max': np.max(features, axis=0),
            'Median': np.median(features, axis=0),
            'Q25': np.percentile(features, 25, axis=0),
            'Q75': np.percentile(features, 75, axis=0),
            'NaN_Count': np.sum(np.isnan(features), axis=0),
            'Inf_Count': np.sum(np.isinf(features), axis=0)
        })
        feature_stats.to_excel(writer, sheet_name='Feature_Statistics', index=False)
        
        # Sheet 6: Feature Statistics by Condition
        feature_stats_by_condition_data = []
        
        # Extract condition information from metadata
        conditions = []
        if metadata:
            for meta in metadata:
                condition = meta.get('condition', 'unknown')
                conditions.append(condition)
        else:
            # Fallback to using labels if no metadata
            conditions = labels
        
        # Get unique conditions
        unique_conditions = list(set(conditions))
        unique_conditions.sort()
        
        # Group by feature first, then by condition for easy comparison
        for i, feature_name in enumerate(feature_names):
            for condition in unique_conditions:
                # Find samples with this condition
                condition_mask = np.array([cond == condition for cond in conditions])
                if np.any(condition_mask):
                    feature_values = features[condition_mask, i]
                    feature_stats_by_condition_data.append({
                        'Feature_Name': feature_name,
                        'Condition': condition,
                        'Mean': np.mean(feature_values),
                        'Std': np.std(feature_values),
                        'Min': np.min(feature_values),
                        'Max': np.max(feature_values),
                        'Median': np.median(feature_values),
                        'Q25': np.percentile(feature_values, 25),
                        'Q75': np.percentile(feature_values, 75),
                        'NaN_Count': np.sum(np.isnan(feature_values)),
                        'Inf_Count': np.sum(np.isinf(feature_values)),
                        'Sample_Count': len(feature_values)
                    })
        
        feature_stats_by_condition_df = pd.DataFrame(feature_stats_by_condition_data)
        
        # Sort by Feature_Name first, then by Condition for easy comparison
        feature_stats_by_condition_df = feature_stats_by_condition_df.sort_values(['Feature_Name', 'Condition'])
        
        feature_stats_by_condition_df.to_excel(writer, sheet_name='Feature_Stats_by_Condition', index=False)
    
    print(f"Features data written to Excel: {output_path}")
    print(f"Dataset info: {features.shape[0]} samples, {features.shape[1]} features")
    print(f"Label distribution: {dict(zip(unique_labels, label_counts))}")
    print(f"Sheets created: Dataset_Info, Feature_Names, Label_Distribution, Features_Data, Feature_Statistics, Feature_Stats_by_Condition")


def write_cv_results_to_excel(cv_results: Dict[str, Dict], output_path: str) -> None:
    """
    Write cross-validation results to Excel format.
    
    Args:
        cv_results: Dictionary mapping frequency to CV results
        output_path: Path to save the Excel file (.xlsx)
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        
        # Sheet 1: Summary
        summary_data = []
        for freq, results in cv_results.items():
            summary_data.append({
                'Frequency': freq,
                'Mean_Accuracy': results['mean_accuracy'],
                'Std_Accuracy': results['std_accuracy'],
                'Best_Fold': results['best_fold'],
                'Worst_Fold': results['worst_fold'],
                'Stability_Score': results['mean_accuracy'] / (results['std_accuracy'] + 1e-8),
                'N_Samples': results['n_samples'],
                'N_Features': results['n_features']
            })
        
        summary_df = pd.DataFrame(summary_data).sort_values('Mean_Accuracy', ascending=False)
        summary_df.to_excel(writer, sheet_name='CV_Summary', index=False)
        
        # Sheet 2: Detailed CV Scores by Fold
        detailed_scores = []
        for freq, results in cv_results.items():
            for i, score in enumerate(results['cv_scores']):
                detailed_scores.append({
                    'Frequency': freq,
                    'Fold': i + 1,
                    'Accuracy': score
                })
        
        detailed_df = pd.DataFrame(detailed_scores)
        detailed_df.to_excel(writer, sheet_name='CV_Scores_Detail', index=False)
        
        # Sheet 3: Label Distribution by Frequency
        label_dist_data = []
        for freq, results in cv_results.items():
            for label, count in results['label_distribution'].items():
                label_dist_data.append({
                    'Frequency': freq,
                    'Class': label,
                    'Count': count,
                    'Percentage': (count / results['n_samples'] * 100)
                })
        
        label_dist_df = pd.DataFrame(label_dist_data)
        label_dist_df.to_excel(writer, sheet_name='Label_Distribution', index=False)
        
        # Sheet 4: Analysis Info
        analysis_info = {
            'Property': [
                'Number of Frequencies Analyzed',
                'Best Performing Frequency',
                'Worst Performing Frequency',
                'Most Stable Frequency',
                'Analysis Date'
            ],
            'Value': [
                len(cv_results),
                max(cv_results.keys(), key=lambda x: cv_results[x]['mean_accuracy']),
                min(cv_results.keys(), key=lambda x: cv_results[x]['mean_accuracy']),
                max(cv_results.keys(), key=lambda x: cv_results[x]['mean_accuracy'] / (cv_results[x]['std_accuracy'] + 1e-8)),
                pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            ]
        }
        pd.DataFrame(analysis_info).to_excel(writer, sheet_name='Analysis_Info', index=False)
    
    print(f"CV results written to Excel: {output_path}")
    print(f"Sheets created: CV_Summary, CV_Scores_Detail, Label_Distribution, Analysis_Info")


def evaluate_model_cv(features: np.ndarray, labels: np.ndarray, 
                     frequency: str, cv_folds: int = 5) -> Dict[str, Any]:
    """
    Evaluate model performance using cross-validation
    
    Args:
        features: Feature matrix
        labels: Labels
        frequency: Frequency label for reporting
        cv_folds: Number of CV folds
        
    Returns:
        Dictionary with CV results
    """
    print(f"Evaluating model for {frequency}...")
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        ))
    ])
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, features, labels, cv=cv, scoring='accuracy')
    
    # Label distribution
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    label_distribution = dict(zip(unique_labels, label_counts))
    
    results = {
        'frequency': frequency,
        'cv_scores': cv_scores,
        'mean_accuracy': cv_scores.mean(),
        'std_accuracy': cv_scores.std(),
        'best_fold': cv_scores.max(),
        'worst_fold': cv_scores.min(),
        'label_distribution': label_distribution,
        'n_samples': len(labels),
        'n_features': features.shape[1]
    }
    
    print(f"{frequency} - Mean CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    return results


def plot_cv_scores_by_fold(cv_results: Dict[str, Dict]) -> None:
    """
    Plot CV scores by fold for each frequency (similar to cell 3 style)
    
    Args:
        cv_results: Dictionary mapping frequency to CV results
    """
    frequencies = list(cv_results.keys())
    n_freq = len(frequencies)
    
    # Calculate subplot layout
    cols = min(2, n_freq)
    rows = (n_freq + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    if n_freq == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if n_freq > 1 else [axes]
    else:
        axes = axes.flatten()
    
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold', 'lightpink', 'lightyellow']
    
    for i, freq in enumerate(frequencies):
        if i >= len(axes):
            break
            
        cv_scores = cv_results[freq]['cv_scores']
        mean_acc = cv_results[freq]['mean_accuracy']
        
        # Bar plot for each fold
        axes[i].bar(range(1, 6), cv_scores, alpha=0.7, color=colors[i % len(colors)])
        
        # Mean line
        axes[i].axhline(y=mean_acc, color='red', linestyle='--', 
                       label=f'Mean: {mean_acc:.3f}', linewidth=2)
        
        axes[i].set_xlabel('Fold')
        axes[i].set_ylabel('Accuracy')
        axes[i].set_title(f'CV Scores by Fold - {freq.upper()}')
        axes[i].set_ylim(0, 1)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for j, score in enumerate(cv_scores):
            axes[i].text(j + 1, score + 0.01, f'{score:.3f}', 
                        ha='center', va='bottom', fontsize=9)
    
    # Hide unused subplots
    for i in range(n_freq, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_cv_results_comparison(cv_results: Dict[str, Dict]) -> None:
    """
    Plot cross-validation results comparison across frequencies
    
    Args:
        cv_results: Dictionary mapping frequency to CV results
    """
    frequencies = list(cv_results.keys())
    n_freq = len(frequencies)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Mean accuracy comparison
    mean_accs = [cv_results[freq]['mean_accuracy'] for freq in frequencies]
    std_accs = [cv_results[freq]['std_accuracy'] for freq in frequencies]
    
    axes[0].bar(frequencies, mean_accs, yerr=std_accs, alpha=0.7, capsize=5)
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Mean CV Accuracy by Frequency')
    axes[0].set_ylim(0, 1)
    
    # 2. Sample size comparison
    n_samples = [cv_results[freq]['n_samples'] for freq in frequencies]
    n_features = [cv_results[freq]['n_features'] for freq in frequencies]
    
    axes[1].bar(frequencies, n_samples, alpha=0.7)
    axes[1].set_ylabel('Number of Samples')
    axes[1].set_title('Dataset Size by Frequency')
    
    # 3. Stability analysis (std vs mean)
    stability_scores = [mean / (std + 1e-8) for mean, std in zip(mean_accs, std_accs)]
    axes[2].bar(frequencies, stability_scores, alpha=0.7)
    axes[2].set_ylabel('Stability Score (Mean/Std)')
    axes[2].set_title('Model Stability by Frequency')
    
    plt.tight_layout()
    plt.show()


def create_performance_summary(cv_results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create a summary DataFrame of performance across frequencies
    
    Args:
        cv_results: Dictionary mapping frequency to CV results
        
    Returns:
        Summary DataFrame
    """
    summary_data = []
    
    for freq, results in cv_results.items():
        summary_data.append({
            'Frequency': freq,
            'Mean_Accuracy': results['mean_accuracy'],
            'Std_Accuracy': results['std_accuracy'],
            'Best_Fold': results['best_fold'],
            'Worst_Fold': results['worst_fold'],
            'Stability_Score': results['mean_accuracy'] / (results['std_accuracy'] + 1e-8),
            'N_Samples': results['n_samples'],
            'N_Features': results['n_features']
        })
    
    summary_df = pd.DataFrame(summary_data).sort_values('Mean_Accuracy', ascending=False)
    
    return summary_df

def run_comprehensive_frequency_analysis(data_loader, frequencies: List[str], 
                                       load: str = "no load", 
                                       max_windows_per_class: int = 20,
                                       window_length: int = 1024,
                                       window_overlap: float = 0.5,
                                       export_to_excel: bool = False,
                                       output_dir: str = "output") -> Dict:
    """
    Run comprehensive CV analysis across multiple frequencies
    
    Args:
        data_loader: DataLoader instance
        frequencies: List of frequencies to analyze
        load: Load condition
        max_windows_per_class: Maximum number of windows per class (default: 20)
        window_length: Window length for analysis
        window_overlap: Window overlap ratio
        export_to_excel: Whether to export CV results to Excel files
        output_dir: Directory to save Excel files
        
    Returns:
        Dictionary of CV results by frequency
    """
    from ml_toolbox.analysis.feature_analysis import extract_features_for_frequency
    
    cv_results = {}
    
    print(f"Starting comprehensive CV analysis for frequencies: {frequencies}")
    print(f"Load condition: {load}")
    print(f"Max windows per class: {max_windows_per_class}")
    if export_to_excel:
        print(f"Excel export enabled - output directory: {output_dir}")
    print("=" * 60)
    
    for freq in frequencies:
        try:
            print(f"Starting feature extraction for {freq}...")
            
            # Extract features
            features, labels, feature_names, metadata = extract_features_for_frequency(
                data_loader, freq, load, max_windows_per_class=max_windows_per_class, 
                window_size=window_length, overlap_ratio=window_overlap
            )
            
            print(f"Feature extraction completed for {freq}. Shape: {features.shape}")
            
            # Export features to Excel if requested
            if export_to_excel:
                features_excel_path = f"{output_dir}/features_{freq}_{load.replace(' ', '_')}.xlsx"
                write_features_to_excel(features, labels, feature_names, features_excel_path, 
                                       frequency=freq, metadata=metadata)
            
            print(f"Starting CV evaluation for {freq}...")
            
            # Evaluate model with cross-validation
            cv_result = evaluate_model_cv(features, labels, freq)
            cv_results[freq] = cv_result
            
            print(f"{freq} CV analysis completed")
            
        except Exception as e:
            import traceback
            print(f"Error analyzing {freq}:")
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception message: {str(e)}")
            print("Full traceback:")
            traceback.print_exc()
            print("-" * 50)
            continue
    
    # Export CV results to Excel if requested
    if export_to_excel and cv_results:
        cv_excel_path = f"{output_dir}/cv_results_{load.replace(' ', '_')}.xlsx"
        write_cv_results_to_excel(cv_results, cv_excel_path)
    
    print("=" * 60)
    print(f"CV analysis completed for {len(cv_results)} frequencies")
    
    return cv_results


def evaluate_incremental_features_cv(features: np.ndarray, labels: np.ndarray, 
                                    feature_names: List[str], 
                                    feature_order: Optional[List[int]] = None,
                                    cv_folds: int = 5,
                                    max_features: Optional[int] = None) -> Dict[str, Any]:
    """
    Iteratively evaluate CV performance by incrementally adding features.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        labels: Labels array (n_samples,)
        feature_names: List of feature names
        feature_order: Optional list of feature indices to use in specific order.
                      If None, uses the original order (0, 1, 2, ...)
        cv_folds: Number of CV folds
        max_features: Maximum number of features to evaluate (if None, uses all)
        
    Returns:
        Dictionary containing:
        - 'results': List of results for each iteration
        - 'best_n_features': Number of features that achieved best performance
        - 'best_accuracy': Best mean accuracy achieved
        - 'feature_importance_progression': How feature importance changes
    """
    print("Starting incremental feature evaluation...")
    
    n_features = features.shape[1]
    
    # Determine feature order
    if feature_order is None:
        feature_order = list(range(n_features))
    else:
        # Validate feature_order
        if max(feature_order) >= n_features or min(feature_order) < 0:
            raise ValueError(f"feature_order contains invalid indices. Valid range: 0-{n_features-1}")
    
    # Determine maximum features to evaluate
    if max_features is None:
        max_features = len(feature_order)
    else:
        max_features = min(max_features, len(feature_order))
    
    print(f"Evaluating incremental feature sets from 1 to {max_features} features")
    print(f"Feature order: {feature_order[:max_features]}")
    print("=" * 60)
    
    results = []
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    best_accuracy = 0
    best_n_features = 1
    
    for n_feat in range(1, max_features + 1):
        try:
            # Select features for this iteration
            selected_indices = feature_order[:n_feat]
            selected_features = features[:, selected_indices]
            selected_feature_names = [feature_names[i] for i in selected_indices]
            
            print(f"Evaluating with {n_feat} feature(s): {selected_feature_names[-1] if n_feat == 1 else f'{selected_feature_names[0]}...{selected_feature_names[-1]}'}")
            
            # Create pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('rf', RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2
                ))
            ])
            
            # Cross-validation
            cv_scores = cross_val_score(pipeline, selected_features, labels, cv=cv, scoring='accuracy')
            
            # Fit model to get feature importance
            pipeline.fit(selected_features, labels)
            feature_importance = pipeline.named_steps['rf'].feature_importances_
            
            mean_accuracy = cv_scores.mean()
            std_accuracy = cv_scores.std()
            
            # Track best performance
            if mean_accuracy > best_accuracy:
                best_accuracy = mean_accuracy
                best_n_features = n_feat
            
            result = {
                'n_features': n_feat,
                'selected_indices': selected_indices,
                'selected_feature_names': selected_feature_names,
                'cv_scores': cv_scores,
                'mean_accuracy': mean_accuracy,
                'std_accuracy': std_accuracy,
                'best_fold': cv_scores.max(),
                'worst_fold': cv_scores.min(),
                'feature_importance': feature_importance,
                'improvement_from_previous': mean_accuracy - results[-1]['mean_accuracy'] if results else 0
            }
            
            results.append(result)
            
            print(f"  → Mean CV Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
            if results and len(results) > 1:
                improvement = result['improvement_from_previous']
                print(f"  → Improvement: {improvement:+.4f}")
            
        except Exception as e:
            print(f"Error evaluating {n_feat} features: {str(e)}")
            continue
    
    print("=" * 60)
    print(f"Incremental evaluation completed")
    print(f"Best performance: {best_accuracy:.4f} with {best_n_features} features")
    
    return {
        'results': results,
        'best_n_features': best_n_features,
        'best_accuracy': best_accuracy,
        'n_samples': len(labels),
        'total_features_available': n_features,
        'feature_order': feature_order[:max_features]
    }


def plot_incremental_feature_performance(incremental_results: Dict[str, Any]) -> None:
    """
    Plot the performance progression as features are added incrementally.
    
    Args:
        incremental_results: Results from evaluate_incremental_features_cv
    """
    results = incremental_results['results']
    
    if not results:
        print("No results to plot")
        return
    
    n_features_list = [r['n_features'] for r in results]
    mean_accuracies = [r['mean_accuracy'] for r in results]
    std_accuracies = [r['std_accuracy'] for r in results]
    improvements = [r['improvement_from_previous'] for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Accuracy progression with error bars
    axes[0, 0].errorbar(n_features_list, mean_accuracies, yerr=std_accuracies, 
                       marker='o', capsize=5, linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Number of Features')
    axes[0, 0].set_ylabel('Mean CV Accuracy')
    axes[0, 0].set_title('CV Accuracy vs Number of Features')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Mark best performance
    best_idx = incremental_results['best_n_features'] - 1
    if best_idx < len(mean_accuracies):
        axes[0, 0].scatter(incremental_results['best_n_features'], 
                          incremental_results['best_accuracy'], 
                          color='red', s=100, marker='*', 
                          label=f'Best: {incremental_results["best_accuracy"]:.4f}')
        axes[0, 0].legend()
    
    # 2. Improvement from previous iteration
    axes[0, 1].bar(n_features_list[1:], improvements[1:], alpha=0.7)
    axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Number of Features')
    axes[0, 1].set_ylabel('Improvement from Previous')
    axes[0, 1].set_title('Performance Improvement per Added Feature')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Standard deviation progression
    axes[1, 0].plot(n_features_list, std_accuracies, marker='s', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Number of Features')
    axes[1, 0].set_ylabel('Standard Deviation')
    axes[1, 0].set_title('Model Stability vs Number of Features')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Feature importance for best model
    if incremental_results['best_n_features'] <= len(results):
        best_result = results[incremental_results['best_n_features'] - 1]
        feature_names = best_result['selected_feature_names']
        importance = best_result['feature_importance']
        
        # Truncate feature names if too long
        display_names = [name[:20] + '...' if len(name) > 20 else name for name in feature_names]
        
        y_pos = np.arange(len(feature_names))
        axes[1, 1].barh(y_pos, importance, alpha=0.7)
        axes[1, 1].set_yticks(y_pos)
        axes[1, 1].set_yticklabels(display_names)
        axes[1, 1].set_xlabel('Feature Importance')
        axes[1, 1].set_title(f'Feature Importance (Best Model: {incremental_results["best_n_features"]} features)')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def write_incremental_results_to_excel(incremental_results: Dict[str, Any], 
                                      output_path: str,
                                      frequency: Optional[str] = None) -> None:
    """
    Write incremental feature evaluation results to Excel.
    
    Args:
        incremental_results: Results from evaluate_incremental_features_cv
        output_path: Path to save the Excel file
        frequency: Optional frequency identifier
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    results = incremental_results['results']
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        
        # Sheet 1: Summary
        summary_data = {
            'Property': [
                'Total Features Available',
                'Features Evaluated',
                'Best Number of Features',
                'Best Accuracy',
                'Number of Samples',
                'Frequency',
                'Analysis Date'
            ],
            'Value': [
                incremental_results['total_features_available'],
                len(results),
                incremental_results['best_n_features'],
                incremental_results['best_accuracy'],
                incremental_results['n_samples'],
                frequency if frequency else 'N/A',
                pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        # Sheet 2: Performance Progression
        progression_data = []
        for result in results:
            progression_data.append({
                'N_Features': result['n_features'],
                'Mean_Accuracy': result['mean_accuracy'],
                'Std_Accuracy': result['std_accuracy'],
                'Best_Fold': result['best_fold'],
                'Worst_Fold': result['worst_fold'],
                'Improvement': result['improvement_from_previous'],
                'Added_Feature': result['selected_feature_names'][-1] if result['selected_feature_names'] else 'N/A'
            })
        
        progression_df = pd.DataFrame(progression_data)
        progression_df.to_excel(writer, sheet_name='Performance_Progression', index=False)
        
        # Sheet 3: Detailed CV Scores
        cv_scores_data = []
        for result in results:
            for fold, score in enumerate(result['cv_scores']):
                cv_scores_data.append({
                    'N_Features': result['n_features'],
                    'Fold': fold + 1,
                    'Accuracy': score
                })
        
        cv_scores_df = pd.DataFrame(cv_scores_data)
        cv_scores_df.to_excel(writer, sheet_name='CV_Scores_Detail', index=False)
        
        # Sheet 4: Feature Selection Order
        feature_order_data = []
        for i, feature_idx in enumerate(incremental_results['feature_order']):
            # Find which result contains this feature
            containing_results = [r for r in results if feature_idx in r['selected_indices']]
            if containing_results:
                first_appearance = min([r['n_features'] for r in containing_results if feature_idx in r['selected_indices']])
                feature_order_data.append({
                    'Selection_Order': i + 1,
                    'Feature_Index': feature_idx,
                    'First_Used_At': first_appearance,
                    'Feature_Name': results[first_appearance-1]['selected_feature_names'][results[first_appearance-1]['selected_indices'].index(feature_idx)]
                })
        
        feature_order_df = pd.DataFrame(feature_order_data)
        feature_order_df.to_excel(writer, sheet_name='Feature_Selection_Order', index=False)

    print(f"Incremental results written to Excel: {output_path}")