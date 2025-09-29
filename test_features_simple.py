"""
Simple test for the feature extraction module (without advanced features).
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


def simple_feature_test():
    """Simple and fast test of feature extraction."""
    print("Simple Feature Extraction Test")
    print("=" * 40)
    
    # 1. Load data
    print("\n1. Loading data...")
    data_loader = DataLoader(Path("data_set"))
    
    # Load a small amount of data
    healthy_data, healthy_meta = data_loader.load_batch(
        condition="healthy",
        sensor_type="current", 
        max_workers=2
    )
    
    if not healthy_data:
        print("❌ No data loaded")
        return
    
    print(f"✓ Loaded {len(healthy_data)} files")
    
    # 2. Create windows
    print("\n2. Creating windows...")
    
    # Use just first file and create small windows
    windows, labels, win_metadata = create_windows_for_ml(
        [healthy_data[0]],  # Just first file
        [healthy_meta[0]],
        window_size=2000,   # 0.2 seconds
        overlap_ratio=0.5,
        balance_classes=False,
        max_windows_per_class=10  # Small number for testing
    )
    
    print(f"✓ Created {windows.shape[0]} windows")
    print(f"  - Window shape: {windows.shape}")
    
    # 3. Basic feature extraction (no advanced features)
    print("\n3. Basic feature extraction...")
    
    config = FeatureConfig(
        sampling_rate=10000,
        time_domain=True,
        frequency_domain=True,
        statistical_moments=True,
        shape_factors=True,
        entropy_features=False,  # Disable expensive features
        spectral_features=True
    )
    
    extractor = FeatureExtractor(config)
    
    # Test single window
    single_features = extractor.extract_features_multichannel(
        windows[0], ["current_a", "current_b"]
    )
    
    print(f"✓ Single window: {len(single_features)} features")
    
    # 4. Batch extraction
    print("\n4. Batch extraction...")
    
    feature_matrix, feature_names = extractor.extract_features_batch(
        windows, ["current_a", "current_b"]
    )
    
    print(f"✓ Batch extraction completed")
    print(f"  - Feature matrix shape: {feature_matrix.shape}")
    print(f"  - Total features: {len(feature_names)}")
    
    # 5. Show feature types
    print("\n5. Feature analysis...")
    
    time_features = [f for f in feature_names if any(x in f for x in ['mean', 'std', 'rms', 'peak'])]
    freq_features = [f for f in feature_names if any(x in f for x in ['spectral', 'frequency', 'band'])]
    stat_features = [f for f in feature_names if any(x in f for x in ['skewness', 'kurtosis'])]
    shape_features = [f for f in feature_names if any(x in f for x in ['crest', 'form', 'impulse'])]
    
    print(f"  - Time domain: {len(time_features)} features")
    print(f"  - Frequency domain: {len(freq_features)} features")  
    print(f"  - Statistical: {len(stat_features)} features")
    print(f"  - Shape factors: {len(shape_features)} features")
    
    # 6. Quality check
    print("\n6. Quality check...")
    
    nan_count = np.sum(np.isnan(feature_matrix))
    inf_count = np.sum(np.isinf(feature_matrix))
    zero_var = np.sum(np.std(feature_matrix, axis=0) < 1e-10)
    
    print(f"  - NaN values: {nan_count}")
    print(f"  - Infinite values: {inf_count}")
    print(f"  - Zero variance features: {zero_var}")
    
    if nan_count == 0 and inf_count == 0:
        print("  ✓ Feature matrix is clean!")
    
    # 7. ML-ready extraction
    print("\n7. ML-ready extraction...")
    
    ml_features, ml_names = extract_features_for_ml(
        windows,
        sampling_rate=10000,
        sensor_type="current",
        metadata_list=win_metadata
    )
    
    print(f"✓ ML-ready features: {ml_features.shape}")
    print(f"  - Features per window: {len(ml_names)}")
    
    print("\n" + "=" * 40)
    print("✅ Feature extraction test successful!")
    print(f"   • {ml_features.shape[1]} features extracted per window")
    print(f"   • {ml_features.shape[0]} windows processed")
    print(f"   • Ready for machine learning!")


if __name__ == "__main__":
    simple_feature_test()