"""
Windowing Module - Quick Reference Guide

This guide provides examples for common windowing tasks for feature extraction
and machine learning with motor health monitoring data.
"""

from pathlib import Path
from ml_toolbox import DataLoader, WindowConfig, create_windows_for_ml, WindowAnalyzer
import numpy as np


def quick_examples():
    """Quick examples for common windowing tasks."""
    
    # 1. Basic Setup
    data_loader = DataLoader(Path("data_set"))
    
    # 2. Load data for windowing
    print("=== Quick Windowing Examples ===\n")
    
    # Example 1: 1-second windows with 50% overlap for feature extraction
    print("1. Feature Extraction Windows (1 second, 50% overlap)")
    
    # Load healthy current data
    healthy_data, healthy_meta = data_loader.load_batch(
        condition="healthy", 
        sensor_type="current",
        max_workers=2
    )
    
    if healthy_data:
        X, y, metadata = create_windows_for_ml(
            healthy_data[:2],  # First 2 files
            healthy_meta[:2],
            window_size=10000,  # 1 second at 10kHz
            overlap_ratio=0.5,
            balance_classes=False
        )
        print(f"   Windows created: {X.shape}")
        print(f"   Time per window: 1.0 seconds")
        print(f"   Total duration: {X.shape[0] * 0.5:.1f} seconds of data\n")
    
    # Example 2: Short windows for transient detection
    print("2. Transient Detection Windows (0.1 second, 75% overlap)")
    
    if healthy_data:
        X_short, y_short, meta_short = create_windows_for_ml(
            healthy_data[:1],  # First file only
            healthy_meta[:1],
            window_size=1000,   # 0.1 second
            overlap_ratio=0.75, # High overlap for transients
            balance_classes=False
        )
        print(f"   Windows created: {X_short.shape}")
        print(f"   Time per window: 0.1 seconds")
        print(f"   Step size: 0.025 seconds\n")
    
    # Example 3: Balanced multi-class dataset
    print("3. Balanced Multi-Class Dataset")
    
    # Load multiple conditions
    conditions = ["healthy", "faulty_bearing", "misalignment"]
    all_data = []
    all_meta = []
    
    for condition in conditions:
        data, meta = data_loader.load_batch(
            condition=condition,
            sensor_type="current",
            max_workers=2
        )
        if data:
            all_data.extend(data[:2])  # 2 files per condition
            all_meta.extend(meta[:2])
    
    if all_data:
        X_balanced, y_balanced, meta_balanced = create_windows_for_ml(
            all_data,
            all_meta,
            window_size=5000,    # 0.5 seconds
            overlap_ratio=0.3,   # 30% overlap
            balance_classes=True,
            max_windows_per_class=30  # Equal representation
        )
        
        print(f"   Windows created: {X_balanced.shape}")
        print(f"   Label distribution: {np.bincount(y_balanced)}")
        
        # Show class names
        class_names = {0: "healthy", 1: "faulty_bearing", 2: "misalignment"}
        for label, count in enumerate(np.bincount(y_balanced)):
            if count > 0:
                print(f"   - {class_names.get(label, f'class_{label}')}: {count} windows")
        print()
    
    # Example 4: Window size optimization
    print("4. Window Size Optimization")
    
    if healthy_data:
        data_length = healthy_data[0].shape[0]
        
        # Analyze different window sizes
        window_sizes = [1000, 2000, 5000, 10000]  # 0.1s to 1s
        
        print(f"   Data length: {data_length} samples ({data_length/10000:.1f} seconds)")
        print("   Window Size | Windows | Coverage | Overlap")
        print("   ------------|---------|----------|--------")
        
        for ws in window_sizes:
            config = WindowConfig(window_size=ws, step_size=ws//2)  # 50% overlap
            analysis = WindowAnalyzer.analyze_windowing(data_length, config)
            
            if analysis['feasible']:
                print(f"   {ws:6d} smp  | {analysis['n_windows']:7d} | {analysis['coverage_ratio']:7.1%} | {analysis['overlap_ratio']:6.1%}")
        print()
    
    # Example 5: Memory-efficient processing for large datasets
    print("5. Memory-Efficient Processing Pattern")
    print("""
    from ml_toolbox import SlidingWindowGenerator, WindowConfig
    
    # For very large datasets, use generator pattern:
    config = WindowConfig(window_size=10000, step_size=5000)
    generator = SlidingWindowGenerator(config)
    
    features = []
    for window, metadata in generator.generate_windows(large_data, meta):
        # Extract features from each window
        feature_vector = extract_features(window)  # Your feature function
        features.append(feature_vector)
    
    # Features extracted without loading all windows into memory
    """)
    
    print("\n=== Common Use Cases ===")
    print("• Fault Detection: 0.5-2 second windows, 50% overlap")
    print("• Transient Analysis: 0.05-0.2 second windows, 75% overlap") 
    print("• Bearing Defects: 1-5 second windows, 30-50% overlap")
    print("• Vibration Analysis: Match bearing rotation period")
    print("• Current Analysis: Match electrical frequency harmonics")
    print("\n=== Feature Extraction Tips ===")
    print("• Time Domain: mean, std, RMS, peak, crest factor, kurtosis")
    print("• Frequency Domain: FFT, PSD, spectral peaks, harmonics")
    print("• Time-Frequency: STFT, wavelets, envelope analysis")
    print("• Statistical: percentiles, moments, entropy")


if __name__ == "__main__":
    quick_examples()