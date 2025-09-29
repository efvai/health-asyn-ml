"""
Test and example usage of the windowing module.
"""

import sys
from pathlib import Path
import numpy as np

# Add the ml_toolbox to path
sys.path.append(str(Path(__file__).parent))

from ml_toolbox import DataLoader, WindowConfig, WindowExtractor, create_windows_for_ml, WindowAnalyzer


def test_windowing():
    """Test the windowing functionality with real motor data."""
    print("Testing Windowing Module")
    print("=" * 50)
    
    # Load some sample data
    print("\n1. Loading Sample Data...")
    data_loader = DataLoader(Path("data_set"))
    
    # Load a small batch of healthy data for testing
    data_list, metadata_list = data_loader.load_batch(
        condition="healthy",
        load="no_load", 
        sensor_type="current",
        max_workers=2
    )
    
    if not data_list:
        print("❌ No data loaded. Please check if data files exist.")
        return
    
    print(f"✓ Loaded {len(data_list)} files")
    print(f"  - First file shape: {data_list[0].shape}")
    print(f"  - Data type: {metadata_list[0]['sensor_type']}")
    
    # 2. Basic Windowing
    print("\n2. Basic Windowing...")
    
    # Configure windows: 1 second windows with 50% overlap
    sampling_rate = 10000  # 10kHz
    window_size = sampling_rate  # 1 second
    overlap_ratio = 0.5
    
    config = WindowConfig(
        window_size=window_size,
        step_size=int(window_size * (1 - overlap_ratio)),
        overlap_ratio=overlap_ratio
    )
    
    print(f"  - Window size: {window_size} samples (1 second)")
    print(f"  - Overlap: {overlap_ratio * 100}%")
    print(f"  - Step size: {config.step_size} samples")
    
    # Test with first file
    test_data = data_list[0]
    test_metadata = metadata_list[0]
    
    extractor = WindowExtractor(config)
    windows, window_metadata = extractor.extract_windows(test_data, test_metadata)
    
    print(f"✓ Extracted {len(windows)} windows from first file")
    print(f"  - Window shape: {windows.shape}")
    print(f"  - Original data: {test_data.shape[0]} samples")
    print(f"  - Coverage: {windows.shape[0] * config.step_size + window_size} samples")
    
    # 3. Window Analysis
    print("\n3. Window Analysis...")
    
    analysis = WindowAnalyzer.analyze_windowing(test_data.shape[0], config)
    print(f"✓ Windowing analysis:")
    print(f"  - Feasible: {analysis['feasible']}")
    print(f"  - Number of windows: {analysis['n_windows']}")
    print(f"  - Coverage ratio: {analysis['coverage_ratio']:.2%}")
    print(f"  - Overlap ratio: {analysis['overlap_ratio']:.2%}")
    print(f"  - Unused samples: {analysis['unused_samples']}")
    print(f"  - Data expansion ratio: {analysis['data_expansion_ratio']:.2f}x")
    
    # 4. Batch Windowing
    print("\n4. Batch Windowing...")
    
    # Use smaller window for demonstration with multiple files
    small_config = WindowConfig(window_size=5000, step_size=3500, overlap_ratio=0.3)  # 0.5 second windows
    small_extractor = WindowExtractor(small_config)
    
    all_windows, all_metadata = small_extractor.extract_windows_batch(
        data_list[:3],  # Use first 3 files only
        metadata_list[:3]
    )
    
    print(f"✓ Batch windowing on {len(data_list[:3])} files:")
    print(f"  - Total windows: {len(all_windows)}")
    print(f"  - Window shape: {all_windows.shape}")
    
    # Show distribution by file
    file_counts = {}
    for meta in all_metadata:
        filename = meta['filename']
        file_counts[filename] = file_counts.get(filename, 0) + 1
    
    for filename, count in file_counts.items():
        print(f"  - {filename}: {count} windows")
    
    # 5. ML-Ready Windows
    print("\n5. ML-Ready Windowing...")
    
    try:
        # Load data from multiple conditions for ML
        healthy_data, healthy_meta = data_loader.load_batch(
            condition="healthy", sensor_type="current", max_workers=2
        )
        faulty_data, faulty_meta = data_loader.load_batch(
            condition="faulty_bearing", sensor_type="current", max_workers=2
        )
        
        if healthy_data and faulty_data:
            # Combine data
            combined_data = healthy_data[:2] + faulty_data[:2]  # 2 files from each
            combined_meta = healthy_meta[:2] + faulty_meta[:2]
            
            # Create ML-ready windows
            X, y, win_meta = create_windows_for_ml(
                combined_data,
                combined_meta,
                window_size=2000,  # 0.2 second windows
                overlap_ratio=0.5,
                balance_classes=True,
                max_windows_per_class=50  # Limit for demo
            )
            
            print(f"✓ ML-ready windows created:")
            print(f"  - X shape: {X.shape}")
            print(f"  - y shape: {y.shape}")
            print(f"  - Label distribution: {np.bincount(y)}")
            
            # Show class distribution
            from collections import Counter
            label_mapping = {0: "healthy", 1: "faulty_bearing"}
            label_dist = Counter(y)
            for label, count in label_dist.items():
                class_name = label_mapping.get(label, f"unknown_{label}")
                print(f"    - {class_name}: {count} windows")
        
        else:
            print("⚠ Could not load both healthy and faulty data for ML demo")
    
    except Exception as e:
        print(f"⚠ ML demo failed: {e}")
    
    # 6. Memory-Efficient Generator
    print("\n6. Memory-Efficient Processing...")
    
    from ml_toolbox import SlidingWindowGenerator
    
    generator = SlidingWindowGenerator(WindowConfig(window_size=1000, step_size=500))
    
    # Process first file with generator (memory efficient)
    window_count = 0
    for window, win_meta in generator.generate_windows(test_data, test_metadata):
        window_count += 1
        if window_count <= 3:  # Show first 3 windows
            print(f"  Window {win_meta['window_id']}: samples {win_meta['start_sample']}-{win_meta['end_sample']}")
    
    print(f"✓ Generated {window_count} windows using memory-efficient generator")
    
    # 7. Window Configuration Suggestions
    print("\n7. Window Configuration Suggestions...")
    
    data_length = test_data.shape[0]
    suggested_config = WindowAnalyzer.suggest_window_config(
        data_length=data_length,
        target_windows=100,  # Want ~100 windows
        overlap_ratio=0.5
    )
    
    print(f"✓ Suggested config for {data_length} samples to get ~100 windows:")
    print(f"  - Window size: {suggested_config.window_size} samples")
    print(f"  - Step size: {suggested_config.step_size} samples")
    print(f"  - Overlap ratio: {suggested_config.overlap_ratio}")
    
    # Verify the suggestion
    verification = WindowAnalyzer.analyze_windowing(data_length, suggested_config)
    print(f"  - Actual windows: {verification['n_windows']}")
    print(f"  - Coverage: {verification['coverage_ratio']:.2%}")
    
    print("\n" + "=" * 50)
    print("Windowing module test completed!")
    print("\nKey Features Demonstrated:")
    print("✓ Basic sliding windows with configurable overlap")
    print("✓ Batch processing of multiple files")
    print("✓ ML-ready data preparation with class balancing")
    print("✓ Memory-efficient generator for large datasets")
    print("✓ Window analysis and configuration suggestions")
    print("✓ Comprehensive metadata tracking")


if __name__ == "__main__":
    test_windowing()