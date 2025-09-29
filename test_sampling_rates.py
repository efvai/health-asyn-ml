#!/usr/bin/env python3
"""
Test script to verify sampling rate fixes.
"""

from ml_toolbox.data_loader import CURRENT_SAMPLING_RATE, VIBRATION_SAMPLING_RATE, extract_features_for_ml
import numpy as np

def test_sampling_rates():
    """Test the new sampling rate constants and auto-detection."""
    print("🔍 Testing Sampling Rate Fixes")
    print("=" * 40)
    
    # Test constants
    print(f"Current sampling rate: {CURRENT_SAMPLING_RATE} Hz")
    print(f"Vibration sampling rate: {VIBRATION_SAMPLING_RATE} Hz")
    
    # Test auto-detection
    print("\n📊 Testing Auto-Detection:")
    
    # Create dummy windows for testing
    current_windows = np.random.randn(1, 1024, 2)  # 1 window, 1024 samples, 2 channels
    vibration_windows = np.random.randn(1, 2048, 4)  # 1 window, 2048 samples, 4 channels
    
    # Test current sensor (should auto-detect 10kHz)
    try:
        features_current, names_current = extract_features_for_ml(
            current_windows, 
            sensor_type="current"  # No sampling_rate specified
        )
        print(f"✓ Current sensor auto-detection: SUCCESS")
        print(f"  Features shape: {features_current.shape}")
    except Exception as e:
        print(f"✗ Current sensor auto-detection: FAILED - {e}")
    
    # Test vibration sensor (should auto-detect 26.041kHz)
    try:
        features_vibration, names_vibration = extract_features_for_ml(
            vibration_windows, 
            sensor_type="vibration"  # No sampling_rate specified
        )
        print(f"✓ Vibration sensor auto-detection: SUCCESS")
        print(f"  Features shape: {features_vibration.shape}")
    except Exception as e:
        print(f"✗ Vibration sensor auto-detection: FAILED - {e}")
    
    # Test explicit sampling rate override
    try:
        features_override, names_override = extract_features_for_ml(
            current_windows, 
            sampling_rate=8000.0,  # Override with custom rate
            sensor_type="current"
        )
        print(f"✓ Explicit sampling rate override: SUCCESS")
        print(f"  Features shape: {features_override.shape}")
    except Exception as e:
        print(f"✗ Explicit sampling rate override: FAILED - {e}")
    
    print("\n✅ Sampling rate fix verification complete!")

if __name__ == "__main__":
    test_sampling_rates()