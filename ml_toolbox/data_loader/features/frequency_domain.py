"""
Frequency-domain feature extraction for motor health monitoring data.

This module provides FFT-based spectral features commonly used
in motor health monitoring and fault diagnosis.
"""

import numpy as np
from typing import Dict, Tuple
from scipy.fft import fft, fftfreq
from scipy.signal import welch
import logging

logger = logging.getLogger(__name__)

class FrequencyDomainFeatures:
    """Extract frequency-domain features from signals."""
    
    @staticmethod
    def _apply_window(signal: np.ndarray, window_type: str = 'hann') -> np.ndarray:
        """Apply windowing function to reduce spectral leakage."""
        if window_type == 'hann':
            window = np.hanning(len(signal))
        elif window_type == 'hamming':
            window = np.hamming(len(signal))
        elif window_type == 'blackman':
            window = np.blackman(len(signal))
        elif window_type == 'none' or window_type is None:
            window = np.ones(len(signal))  # No windowing
        else:
            # Default to Hann window for unknown types
            window = np.hanning(len(signal))
        
        return signal * window
    
    @staticmethod
    def fft_features(signal: np.ndarray, sampling_rate: float, window_type: str = 'hann') -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """
        Extract FFT-based features with windowing to reduce spectral leakage.
        
        Args:
            signal: Input signal array
            sampling_rate: Sampling rate in Hz
            window_type: Window function type ('hann', 'hamming', 'blackman', 'none'). Default is 'hann'
        
        Returns:
            Tuple of (features_dict, fft_magnitude, fft_frequencies)
        """
        features = {}
        
        # Apply windowing to reduce spectral leakage
        #windowed_signal = FrequencyDomainFeatures._apply_window(signal, window_type)
        # Compute FFT
        #fft_vals = np.array(fft(windowed_signal))
        #freqs = fftfreq(len(windowed_signal), 1/sampling_rate)
        
        # Try to welch
        fft_freqs, fft_magnitude = welch(signal, fs=sampling_rate, nperseg=2048)

        # Only positive frequencies
        #n_positive = len(freqs) // 2
        #fft_magnitude = np.abs(fft_vals[:n_positive])
        #fft_freqs = freqs[:n_positive]
        
        # Normalize FFT magnitude
        total_energy = np.sum(fft_magnitude**2)
        if total_energy > 1e-12:
            # Normalize by total spectral energy
            fft_magnitude_normalized = fft_magnitude / np.sqrt(total_energy)
        else:
            fft_magnitude_normalized = fft_magnitude
                
        # Spectral features
        features['spectral_centroid'] = np.sum(fft_freqs * fft_magnitude_normalized) / (np.sum(fft_magnitude_normalized) + 1e-12)
        features['spectral_spread'] = np.sqrt(np.sum((fft_freqs - features['spectral_centroid'])**2 * fft_magnitude_normalized) / (np.sum(fft_magnitude_normalized) + 1e-12))
        
        # Spectral energy
        features['spectral_rolloff'] = FrequencyDomainFeatures._spectral_rolloff(fft_magnitude, fft_freqs, 0.85)
        features['spectral_energy'] = np.sum(fft_magnitude**2)
        
        # Spectral entropy - measure of spectral complexity/randomness
        power_spectrum = fft_magnitude**2
        power_spectrum_norm = power_spectrum / (np.sum(power_spectrum) + 1e-12)
        power_spectrum_norm = power_spectrum_norm[power_spectrum_norm > 1e-12]  # Remove near-zeros
        if len(power_spectrum_norm) > 0:
            features['spectral_entropy'] = float(-np.sum(power_spectrum_norm * np.log2(power_spectrum_norm + 1e-12)))
        else:
            features['spectral_entropy'] = 0.0
        
        # Spectral flatness (Wiener entropy) - measure of how noise-like vs tone-like the spectrum is
        # Ratio of geometric mean to arithmetic mean of power spectrum
        power_spectrum_positive = power_spectrum[power_spectrum > 1e-12]
        if len(power_spectrum_positive) > 0:
            geometric_mean = np.exp(np.mean(np.log(power_spectrum_positive + 1e-12)))
            arithmetic_mean = np.mean(power_spectrum_positive)
            features['spectral_flatness'] = float(geometric_mean / (arithmetic_mean + 1e-12))
        else:
            features['spectral_flatness'] = 0.0
        
        return features, fft_magnitude, fft_freqs
    
    @staticmethod
    def _spectral_rolloff(magnitude: np.ndarray, freqs: np.ndarray, threshold: float = 0.85) -> float:
        """Calculate spectral rolloff frequency."""
        total_energy = np.sum(magnitude**2)
        cumulative_energy = np.cumsum(magnitude**2)
        rolloff_idx = np.where(cumulative_energy >= threshold * total_energy)[0]
        return float(freqs[rolloff_idx[0]]) if len(rolloff_idx) > 0 else float(freqs[-1])
    
    @staticmethod
    def spectral_analysis_features(spectrum: Dict) -> Dict[str, float]:
        """
        Extract spectral analysis features from a spectrum dictionary.
        
        Args:
            spectrum: Dictionary with 'freqs' and 'magnitude' keys
            
        Returns:
            Dictionary of spectral features
        """
        features = {}
        freqs = spectrum['freqs']
        magnitude = spectrum['magnitude']
        
        # Spectral centroid and spread
        total_mag = np.sum(magnitude) + 1e-12
        features['centroid'] = np.sum(freqs * magnitude) / total_mag
        features['spread'] = np.sqrt(np.sum((freqs - features['centroid'])**2 * magnitude) / total_mag)
        
        # Spectral entropy - measure of spectral complexity/randomness
        power_spectrum = magnitude**2
        power_spectrum_norm = power_spectrum / (np.sum(power_spectrum) + 1e-12)
        power_spectrum_norm = power_spectrum_norm[power_spectrum_norm > 1e-12]  # Remove near-zeros
        if len(power_spectrum_norm) > 0:
            features['entropy'] = float(-np.sum(power_spectrum_norm * np.log2(power_spectrum_norm + 1e-12)))
        else:
            features['entropy'] = 0.0

        # Spectral flatness (Wiener entropy) - measure of how noise-like vs tone-like the spectrum is
        # Ratio of geometric mean to arithmetic mean of power spectrum
        power_spectrum_positive = power_spectrum[power_spectrum > 1e-12]
        if len(power_spectrum_positive) > 0:
            geometric_mean = np.exp(np.mean(np.log(power_spectrum_positive + 1e-12)))
            arithmetic_mean = np.mean(power_spectrum_positive)
            features['flatness'] = float(geometric_mean / (arithmetic_mean + 1e-12))
        else:
            features['flatness'] = 0.0
            
        return features
    
    @staticmethod
    def peak_analysis_features(spectrum: Dict, peaks: Dict, peak_cutoff: float = 200.0) -> Dict[str, float]:
        """
        Extract peak analysis features from spectrum and detected peaks.
        
        Args:
            spectrum: Dictionary with 'freqs' and 'magnitude' keys
            peaks: Dictionary with 'peak_freqs', 'peak_magnitudes', 'peak_indices' keys
            peak_cutoff: Frequency cutoff for peak analysis (Hz)
            
        Returns:
            Dictionary of peak analysis features
        """
        features = {}
        peak_count = len(peaks["peak_indices"])
        
        if peak_count > 0:
            peak_freqs = peaks["peak_freqs"]
            peak_mags = peaks["peak_magnitudes"]
            
            total_power = np.sum(peak_mags**2)
            features["peak_power_mean"] = float(np.mean(peak_mags))
            features["peak_power_std"] = float(np.std(peak_mags))

            # Dominant peak relative power
            dom_idx = np.argmax(peak_mags)
            features["dom_rel_peak_power"] = float(peak_mags[dom_idx]**2 / (total_power + 1e-12))
            
            if peak_count > 1:
                # Peak spacing analysis
                sorted_peak_freqs = np.sort(peak_freqs)
                peak_spacing = np.diff(sorted_peak_freqs)
                features["peak_sp_mean"] = float(np.mean(peak_spacing))
                features["peak_sp_std"] = float(np.std(peak_spacing))
                
                # Peak frequency coefficient of variation
                mean_peak_freq = np.mean(peak_freqs)
                if mean_peak_freq > 1e-12:
                    features["peak_freq_cv"] = float(np.std(peak_freqs)) / mean_peak_freq
                else:
                    features["peak_freq_cv"] = 0.0
            else:
                features["peak_sp_mean"] = 0.0
                features["peak_sp_std"] = 0.0
                features["peak_freq_cv"] = 0.0
        else:
            features["peak_power_mean"] = 0.0
            features["peak_power_std"] = 0.0
            features["dom_rel_peak_power"] = 0.0
            features["peak_sp_mean"] = 0.0
            features["peak_sp_std"] = 0.0
            features["peak_freq_cv"] = 0.0
            
        return features
    
    @staticmethod
    def harmonic_analysis_features(spectrum: Dict, peaks: Dict, f0: float, 
                                 max_harmonics: int = 5, tolerance_factor: float = 0.1) -> Dict[str, float]:
        """
        Compute THD-like harmonic analysis features using detected peaks.
        
        Args:
            spectrum: Spectrum dictionary with 'freqs' and 'magnitude'
            peaks: Peaks dictionary with 'peak_freqs' and 'peak_magnitudes'
            f0: Fundamental frequency in Hz
            max_harmonics: Maximum number of harmonics to analyze
            tolerance_factor: Relative tolerance for harmonic matching (e.g., 0.1 = 10%)
            
        Returns:
            Dictionary of harmonic features
        """
        features = {}
        
        freqs = spectrum['freqs']
        magnitude = spectrum['magnitude']
        power_spectrum = magnitude**2
        total_power = np.sum(power_spectrum) + 1e-12
        
        peak_freqs = peaks['peak_freqs']
        peak_mags = peaks['peak_magnitudes']
        peak_powers = peak_mags**2
        
        # Find harmonics by matching peaks to harmonic frequencies
        harmonic_powers = []
        harmonic_freqs = []
        found_harmonics = []
        
        for h in range(1, max_harmonics + 1):
            target_freq = h * f0
            
            # Skip if target frequency is beyond spectrum range
            if target_freq > freqs[-1]:
                break
                
            # Find peak closest to harmonic frequency within tolerance
            tolerance = tolerance_factor * target_freq
            freq_diffs = np.abs(peak_freqs - target_freq)
            closest_idx = np.argmin(freq_diffs)
            
            if freq_diffs[closest_idx] <= tolerance:
                # Found a peak close enough to be considered a harmonic
                harmonic_power = peak_powers[closest_idx]
                harmonic_freq = peak_freqs[closest_idx]
                
                harmonic_powers.append(harmonic_power)
                harmonic_freqs.append(harmonic_freq)
                found_harmonics.append(h)
        
        # Compute harmonic analysis features
        if len(harmonic_powers) > 0:
            total_harmonic_power = np.sum(harmonic_powers)
            fundamental_power = harmonic_powers[0] if 1 in found_harmonics else 0.0
            
            # THD calculations
            thd_power_frac = total_harmonic_power / total_power
            features["thd_power_frac"] = float(thd_power_frac)
            features["harmonic_ratio"] = float(total_harmonic_power / (np.sum(peak_powers) + 1e-12))
            features["fundamental_power_ratio"] = float(fundamental_power / total_power)
            features["harmonic_count"] = float(len(found_harmonics))
                
        else:
            # No harmonics found
            features["thd_power_frac"] = 0.0
            features["harmonic_ratio"] = 0.0
            features["fundamental_power_ratio"] = 0.0
            features["harmonic_count"] = 0.0
            
        return features
    
    @staticmethod
    def cross_spectral_features(spectrum1: Dict, spectrum2: Dict, 
                              ch1_name: str = "ch1", ch2_name: str = "ch2") -> Dict[str, float]:
        """
        Extract cross-spectral features between two spectra.
        
        Args:
            spectrum1: First spectrum dictionary with 'freqs' and 'magnitude'
            spectrum2: Second spectrum dictionary with 'freqs' and 'magnitude'
            ch1_name: Name of first channel
            ch2_name: Name of second channel
            
        Returns:
            Dictionary of cross-spectral features
        """
        features = {}
        
        # Spectral correlation
        min_len = min(len(spectrum1['magnitude']), len(spectrum2['magnitude']))
        if min_len > 1:
            corr_coef = np.corrcoef(spectrum1['magnitude'][:min_len], spectrum2['magnitude'][:min_len])[0, 1]
            features[f"{ch1_name}_{ch2_name}_spectral_corr"] = float(corr_coef) if not np.isnan(corr_coef) else 0.0
        else:
            features[f"{ch1_name}_{ch2_name}_spectral_corr"] = 0.0
        
        # Energy comparison
        total_energy1 = np.sum(spectrum1['magnitude']**2)
        total_energy2 = np.sum(spectrum2['magnitude']**2)
        features[f"{ch1_name}_{ch2_name}_energy_ratio"] = float(total_energy1 / (total_energy2 + 1e-12))
        
        # Energy imbalance
        energy_imbalance = abs(total_energy1 - total_energy2) / (total_energy1 + total_energy2 + 1e-12)
        features[f"{ch1_name}_{ch2_name}_energy_imbalance"] = float(energy_imbalance)
        
        return features
    
    @staticmethod
    def peak_magnitude_comparison(peaks1: Dict, peaks2: Dict, 
                                ch1_name: str = "ch1", ch2_name: str = "ch2") -> Dict[str, float]:
        """
        Compare peak magnitudes between two sets of detected peaks.
        
        Args:
            peaks1: First peaks dictionary with 'peak_magnitudes'
            peaks2: Second peaks dictionary with 'peak_magnitudes'
            ch1_name: Name of first channel
            ch2_name: Name of second channel
            
        Returns:
            Dictionary of peak magnitude comparison features
        """
        features = {}
        
        peak_count1 = len(peaks1.get("peak_indices", []))
        peak_count2 = len(peaks2.get("peak_indices", []))
        
        if peak_count1 > 0 and peak_count2 > 0:
            peak_mags1 = peaks1["peak_magnitudes"]
            peak_mags2 = peaks2["peak_magnitudes"]
            
            # Mean magnitude ratio
            features[f"{ch1_name}_{ch2_name}_mean_mag_ratio"] = float(np.mean(peak_mags1) / (np.mean(peak_mags2) + 1e-12))
        else:
            features[f"{ch1_name}_{ch2_name}_mean_mag_ratio"] = 1.0
            
        return features