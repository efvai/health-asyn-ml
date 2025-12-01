"""
Base classes and configuration for feature extraction.
"""

from dataclasses import dataclass

# Sensor-specific sampling rates
CURRENT_SAMPLING_RATE = 10000.0   # LTR11 - Current sensors
VIBRATION_SAMPLING_RATE = 26041.0 # LTR22 - Vibration sensors
ENV_CARRIER_FREQUENCY = 1670.0    # Hz - Expected carrier frequency for Hilbert envelope analysis

@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    sampling_rate: float = CURRENT_SAMPLING_RATE  # Default for current sensors (LTR11)
    
    # Time domain features
    time_domain: bool = True

    # Frequency domain features
    frequency_domain: bool = False

    # Hilbert envelope features
    hilbert_envelope: bool = True

    # Cross Channel
    cross_channel: bool = True  # Compute cross-channel features (e.g., correlation)
       
    # Windowing parameters for spectral leakage reduction
    window_type: str = 'hann'  # Options: 'hann', 'hamming', 'blackman', 'bartlett', 'kaiser', 'none'