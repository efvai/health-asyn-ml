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
from typing import Dict, List, Tuple, Any


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


def analyze_frequency_performance_trends(cv_results: Dict[str, Dict]) -> None:
    """
    Analyze performance trends across frequencies
    
    Args:
        cv_results: Dictionary mapping frequency to CV results
    """
    # Extract numerical frequency values for trend analysis
    freq_values = []
    accuracies = []
    stabilities = []
    
    for freq, results in cv_results.items():
        freq_num = int(freq.replace('hz', ''))
        freq_values.append(freq_num)
        accuracies.append(results['mean_accuracy'])
        stabilities.append(results['mean_accuracy'] / (results['std_accuracy'] + 1e-8))
    
    # Plot trends
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy trend
    axes[0].plot(freq_values, accuracies, 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Mean CV Accuracy')
    axes[0].set_title('Accuracy vs Frequency')
    axes[0].grid(True, alpha=0.3)
    
    # Stability trend
    axes[1].plot(freq_values, stabilities, 's-', linewidth=2, markersize=8, color='orange')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Stability Score')
    axes[1].set_title('Model Stability vs Frequency')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print insights
    best_freq_idx = np.argmax(accuracies)
    most_stable_idx = np.argmax(stabilities)
    
    print(f"\nFREQUENCY ANALYSIS INSIGHTS:")
    print(f"Best accuracy: {freq_values[best_freq_idx]}Hz ({accuracies[best_freq_idx]:.3f})")
    print(f"Most stable: {freq_values[most_stable_idx]}Hz (stability: {stabilities[most_stable_idx]:.2f})")
    
    if len(freq_values) > 1:
        acc_trend = "increasing" if accuracies[-1] > accuracies[0] else "decreasing"
        print(f"Accuracy trend: {acc_trend} with frequency")


def run_comprehensive_frequency_analysis(data_loader, frequencies: List[str], 
                                       load: str = "no load", 
                                       max_windows_per_class: int = 20,
                                       window_length: int = 1024,
                                       window_overlap: float = 0.5) -> Tuple[Dict, Dict]:
    """
    Run comprehensive analysis across multiple frequencies
    
    Args:
        data_loader: DataLoader instance
        frequencies: List of frequencies to analyze
        load: Load condition
        max_windows_per_class: Maximum number of windows per class (default: 20)
        
    Returns:
        Tuple of (cv_results, importance_results)
    """
    from ml_toolbox.analysis.feature_analysis import (
        extract_features_for_frequency, 
        analyze_feature_importance
    )
    
    cv_results = {}
    importance_results = {}
    
    print(f"Starting comprehensive analysis for frequencies: {frequencies}")
    print(f"Load condition: {load}")
    print(f"Max windows per class: {max_windows_per_class}")
    print("=" * 60)
    
    for freq in frequencies:
        try:
            # Extract features
            features, labels, feature_names, metadata = extract_features_for_frequency(
                data_loader, freq, load, max_windows_per_class=max_windows_per_class, 
                window_size=window_length, overlap_ratio=window_overlap
            )
            
            # Evaluate model
            cv_result = evaluate_model_cv(features, labels, freq)
            cv_results[freq] = cv_result
            
            # Analyze feature importance
            importance_result = analyze_feature_importance(features, labels, feature_names, freq)
            importance_results[freq] = importance_result
            
            print(f"{freq} analysis completed")
            
        except Exception as e:
            print(f"Error analyzing {freq}: {str(e)}")
            continue
    
    print("=" * 60)
    print(f"Analysis completed for {len(cv_results)} frequencies")
    
    return cv_results, importance_results