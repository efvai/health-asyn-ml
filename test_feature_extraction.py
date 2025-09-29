"""
Test and example usage of the feature extraction module.
"""

import sys
from pathlib import Path
import numpy as np

# Add the ml_toolbox to path
sys.path.append(str(Path(__file__).parent))

from ml_toolbox import (
    DataLoader, create_windows_for_ml, 
    FeatureConfig, FeatureExtractor, extract_features_for_ml
)


def test_feature_extraction():
    """Test the feature extraction functionality with real motor data."""
    print("Testing Feature Extraction Module")
    print("=" * 60)
    
    # 1. Load and prepare data
    print("\n1. Loading and Preparing Data...")
    data_loader = DataLoader(Path("data_set"))
    
    # Load healthy and faulty data
    healthy_data, healthy_meta = data_loader.load_batch(
        condition="healthy",
        sensor_type="current", 
        max_workers=2
    )
    
    faulty_data, faulty_meta = data_loader.load_batch(
        condition="faulty_bearing",
        sensor_type="current",
        max_workers=2
    )
    
    if not healthy_data or not faulty_data:
        print("❌ Could not load both healthy and faulty data")
        return
    
    print(f"✓ Loaded {len(healthy_data)} healthy and {len(faulty_data)} faulty files")
    
    # 2. Create windows for feature extraction
    print("\n2. Creating Windows...")
    
    # Combine first 2 files from each condition
    combined_data = healthy_data[:2] + faulty_data[:2]
    combined_meta = healthy_meta[:2] + faulty_meta[:2]
    
    # Create windows (0.5 second windows with 50% overlap)
    windows, labels, win_metadata = create_windows_for_ml(
        combined_data,
        combined_meta,
        window_size=5000,  # 0.5 seconds at 10kHz
        overlap_ratio=0.5,
        balance_classes=True,
        max_windows_per_class=20  # Limit for demo
    )
    
    print(f"✓ Created {windows.shape[0]} windows")
    print(f"  - Window shape: {windows.shape}")
    print(f"  - Label distribution: {np.bincount(labels)}")
    
    # 3. Basic Feature Extraction
    print("\n3. Basic Feature Extraction...")
    
    # Test single channel feature extraction
    single_channel = windows[0, :, 0]  # First window, first channel
    
    config = FeatureConfig(
        sampling_rate=10000,
        time_domain=True,
        frequency_domain=True,
        statistical_moments=True,
        shape_factors=True
    )
    
    extractor = FeatureExtractor(config)
    features = extractor.extract_features(single_channel, "test_channel")
    
    print(f"✓ Extracted {len(features)} features from single channel")
    print("  Sample features:")
    for i, (name, value) in enumerate(list(features.items())[:10]):
        print(f"    - {name}: {value:.4f}")
    if len(features) > 10:
        print(f"    ... and {len(features) - 10} more features")
    
    # 4. Multi-channel Feature Extraction
    print("\n4. Multi-channel Feature Extraction...")
    
    # Test multi-channel (current has 2 channels)
    multi_features = extractor.extract_features_multichannel(
        windows[0], ["current_a", "current_b"]
    )
    
    print(f"✓ Extracted {len(multi_features)} features from multi-channel signal")
    print("  Feature categories:")
    
    # Count features by category
    categories = {}
    for feature_name in multi_features.keys():
        if 'current_a' in feature_name:
            category = 'Channel A'
        elif 'current_b' in feature_name:
            category = 'Channel B'
        elif 'correlation' in feature_name or 'phase_diff' in feature_name:
            category = 'Cross-channel'
        else:
            category = 'Other'
        
        categories[category] = categories.get(category, 0) + 1
    
    for category, count in categories.items():
        print(f"    - {category}: {count} features")
    
    # 5. Batch Feature Extraction
    print("\n5. Batch Feature Extraction...")
    
    # Extract features from all windows
    feature_matrix, feature_names = extractor.extract_features_batch(
        windows, ["current_a", "current_b"]
    )
    
    print(f"✓ Batch extraction completed")
    print(f"  - Feature matrix shape: {feature_matrix.shape}")
    print(f"  - Features per window: {len(feature_names)}")
    print(f"  - Total features extracted: {feature_matrix.size}")
    
    # 6. Analyze Feature Types
    print("\n6. Feature Analysis...")
    
    # Categorize features by type
    feature_types = {
        'Time Domain': 0,
        'Frequency Domain': 0,
        'Statistical': 0,
        'Shape Factors': 0,
        'Cross-channel': 0
    }
    
    for name in feature_names:
        if any(x in name for x in ['mean', 'std', 'rms', 'peak', 'energy', 'power']):
            feature_types['Time Domain'] += 1
        elif any(x in name for x in ['spectral', 'frequency', 'fft', 'band']):
            feature_types['Frequency Domain'] += 1
        elif any(x in name for x in ['skewness', 'kurtosis', 'moment']):
            feature_types['Statistical'] += 1
        elif any(x in name for x in ['crest', 'form', 'impulse', 'clearance', 'shape']):
            feature_types['Shape Factors'] += 1
        elif any(x in name for x in ['correlation', 'phase_diff']):
            feature_types['Cross-channel'] += 1
    
    print("  Feature distribution by type:")
    for ftype, count in feature_types.items():
        print(f"    - {ftype}: {count} features")
    
    # 7. ML-Ready Feature Extraction
    print("\n7. ML-Ready Feature Extraction...")
    
    # Use convenience function
    ml_features, ml_feature_names = extract_features_for_ml(
        windows, 
        sampling_rate=10000,
        sensor_type="current"
    )
    
    print(f"✓ ML-ready features extracted")
    print(f"  - Feature matrix shape: {ml_features.shape}")
    print(f"  - Feature names: {len(ml_feature_names)}")
    
    # 8. Feature Statistics
    print("\n8. Feature Statistics...")
    
    # Calculate basic statistics for features
    feature_means = np.mean(ml_features, axis=0)
    feature_stds = np.std(ml_features, axis=0)
    
    print("  Feature statistics (first 10 features):")
    for i in range(min(10, len(ml_feature_names))):
        print(f"    - {ml_feature_names[i]}: μ={feature_means[i]:.4f}, σ={feature_stds[i]:.4f}")
    
    # Check for problematic features
    zero_variance = np.sum(feature_stds < 1e-10)
    infinite_values = np.sum(np.isinf(ml_features))
    nan_values = np.sum(np.isnan(ml_features))
    
    print(f"  Feature quality check:")
    print(f"    - Zero variance features: {zero_variance}")
    print(f"    - Infinite values: {infinite_values}")
    print(f"    - NaN values: {nan_values}")
    
    # 9. Discriminative Power Analysis
    print("\n9. Discriminative Power Analysis...")
    
    # Simple analysis of feature separability between classes
    healthy_mask = labels == 0
    faulty_mask = labels == 1
    
    if np.sum(healthy_mask) > 0 and np.sum(faulty_mask) > 0:
        healthy_features = ml_features[healthy_mask]
        faulty_features = ml_features[faulty_mask]
        
        # Calculate separability for each feature
        separabilities = []
        for i in range(ml_features.shape[1]):
            h_mean = np.mean(healthy_features[:, i])
            f_mean = np.mean(faulty_features[:, i])
            h_std = np.std(healthy_features[:, i])
            f_std = np.std(faulty_features[:, i])
            
            # Simple separability metric (difference in means / average std)
            avg_std = (h_std + f_std) / 2
            if avg_std > 1e-10:
                separability = abs(h_mean - f_mean) / avg_std
            else:
                separability = 0.0
            
            separabilities.append(separability)
        
        # Find most discriminative features
        top_indices = np.argsort(separabilities)[-10:][::-1]
        
        print("  Top 10 most discriminative features:")
        for i, idx in enumerate(top_indices):
            print(f"    {i+1:2d}. {ml_feature_names[idx]}: {separabilities[idx]:.3f}")
    
    # 10. Advanced Features Demo
    print("\n10. Advanced Features Demo...")
    
    # Test with advanced features enabled
    advanced_config = FeatureConfig(
        sampling_rate=10000,
        time_domain=True,
        frequency_domain=True,
        statistical_moments=True,
        shape_factors=True,
        entropy_features=True,
        spectral_features=True
    )
    
    advanced_extractor = FeatureExtractor(advanced_config)
    
    # Extract from a subset for speed
    test_windows = windows[:5]  # First 5 windows
    advanced_features, advanced_names = advanced_extractor.extract_features_batch(
        test_windows, ["current_a", "current_b"]
    )
    
    print(f"✓ Advanced features extracted")
    print(f"  - Advanced feature count: {len(advanced_names)}")
    print(f"  - Increase over basic: {len(advanced_names) - len(feature_names)} features")
    
    # Show entropy features
    entropy_features = [name for name in advanced_names if 'entropy' in name]
    if entropy_features:
        print(f"  - Entropy features: {len(entropy_features)}")
        for ef in entropy_features[:3]:
            print(f"    • {ef}")
    
    print("\n" + "=" * 60)
    print("Feature Extraction Test Completed!")
    print("\n✅ Key Achievements:")
    print(f"   • Extracted {len(ml_feature_names)} features per window")
    print(f"   • Processed {ml_features.shape[0]} windows successfully")
    print(f"   • Feature quality: {ml_features.shape[0] * ml_features.shape[1] - nan_values - infinite_values} valid values")
    print(f"   • Identified top discriminative features")
    print(f"   • Ready for ML training and analysis!")


if __name__ == "__main__":
    test_feature_extraction()