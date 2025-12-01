import numpy as np
from scipy.signal import butter, filtfilt, hilbert, resample_poly
from typing import Dict, Tuple, Optional, Union, List, Any
from dataclasses import dataclass

@dataclass
class EnvelopeConfig:
    """Configuration for envelope analysis"""
    bandpass_low: float = 3050.0
    bandpass_high: float = 3150.0
    lowpass_cutoff: float = 200.0
    filter_order: int = 4
    decimation_factor: int = 5
    sampling_rate: float = 10000.0
    
    @property
    def nyquist(self) -> float:
        return self.sampling_rate / 2
    
    @property
    def envelope_fs(self) -> float:
        return self.sampling_rate / self.decimation_factor
    
class HilbertEnvelopeAnalyzer:
    """
    Hilbert envelope analyzer with access to intermediate processing stages.
    Useful for both feature extraction and research analysis.
    """
    
    def __init__(self, config: EnvelopeConfig):
        self.config = config
        self._setup_filters()
    
    def _setup_filters(self):
        """Pre-compute filter coefficients"""
        nyquist = self.config.nyquist
        
        # Bandpass filter for carrier frequency extraction
        self.bp_b, self.bp_a = butter(
            self.config.filter_order,
            [self.config.bandpass_low/nyquist, self.config.bandpass_high/nyquist],
            btype='band',
            analog=False
        )
        
        # Lowpass filter for envelope smoothing
        self.lp_b, self.lp_a = butter(
            self.config.filter_order,
            self.config.lowpass_cutoff / nyquist,
            btype='low',
            analog=False
        )
    
    def extract_envelope_with_stages(self, signal: np.ndarray, return_stages: bool = True) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Extract Hilbert envelope with optional access to intermediate stages.
        
        Parameters:
        -----------
        signal : np.ndarray
            Input signal
        return_stages : bool
            If True, returns dict with all intermediate stages
            If False, returns only final envelope
        
        Returns:
        --------
        Union[np.ndarray, Dict[str, np.ndarray]]
            If return_stages=False: Final envelope signal
            If return_stages=True: Dictionary with keys:
                - 'original': Original input signal
                - 'bandpass_filtered': After bandpass filtering
                - 'envelope_raw': Raw Hilbert envelope
                - 'envelope_filtered': Lowpass filtered envelope
                - 'envelope_decimated': Final decimated envelope
        """
        stages = {} if return_stages else None
        
        # Store original
        if return_stages:
            stages['original'] = signal.copy()
        
        # Step 1: Bandpass filtering
        bandpass_filtered = filtfilt(self.bp_b, self.bp_a, signal)
        if return_stages:
            stages['bandpass_filtered'] = bandpass_filtered
        
        # Step 2: Hilbert envelope
        envelope_raw = np.abs(hilbert(bandpass_filtered))
        if return_stages:
            stages['envelope_raw'] = envelope_raw
        
        # Step 3: Lowpass filtering of envelope
        envelope_filtered = filtfilt(self.lp_b, self.lp_a, envelope_raw)
        if return_stages:
            stages['envelope_filtered'] = envelope_filtered
        
        # Step 4: Decimation
        envelope_decimated = resample_poly(
            envelope_filtered, 
            up=1, 
            down=self.config.decimation_factor
        )
        if return_stages:
            stages['envelope_decimated'] = envelope_decimated
        
        return stages if return_stages else envelope_decimated
    
    def extract_envelope(self, signal: np.ndarray) -> np.ndarray:
        """
        Extract final envelope signal only (for production use).
        
        Parameters:
        -----------
        signal : np.ndarray
            Input signal
            
        Returns:
        --------
        np.ndarray
            Final envelope signal
        """
        return self.extract_envelope_with_stages(signal, return_stages=False)
    
    def batch_extract_envelopes(self, signals: list, return_stages: bool = False) -> list:
        """
        Extract envelopes from multiple signals.
        
        Parameters:
        -----------
        signals : list
            List of input signals
        return_stages : bool
            Whether to return intermediate stages
            
        Returns:
        --------
        list
            List of envelopes or stage dictionaries
        """
        return [self.extract_envelope_with_stages(signal, return_stages) for signal in signals]