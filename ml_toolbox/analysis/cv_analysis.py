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
    
    print(f"Features data written to Excel: {output_path}")
    print(f"Dataset info: {features.shape[0]} samples, {features.shape[1]} features")
    print(f"Label distribution: {dict(zip(unique_labels, label_counts))}")
    print(f"Sheets created: Dataset_Info, Feature_Names, Label_Distribution, Features_Data, Feature_Statistics")


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
            # Extract features
            features, labels, feature_names, metadata = extract_features_for_frequency(
                data_loader, freq, load, max_windows_per_class=max_windows_per_class, 
                window_size=window_length, overlap_ratio=window_overlap
            )
            
            # Evaluate model with cross-validation
            cv_result = evaluate_model_cv(features, labels, freq)
            cv_results[freq] = cv_result
            
            print(f"{freq} CV analysis completed")
            
        except Exception as e:
            print(f"Error analyzing {freq}: {str(e)}")
            continue
    
    # Export CV results to Excel if requested
    if export_to_excel and cv_results:
        cv_excel_path = f"{output_dir}/cv_results_{load.replace(' ', '_')}.xlsx"
        write_cv_results_to_excel(cv_results, cv_excel_path)
    
    print("=" * 60)
    print(f"CV analysis completed for {len(cv_results)} frequencies")
    
    return cv_results