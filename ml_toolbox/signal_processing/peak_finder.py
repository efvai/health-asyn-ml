"""Spectral peak finding and harmonic analysis utilities."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def find_spectral_peaks(
    freqs: np.ndarray,
    amplitudes: np.ndarray,
    prominence: float = 0.02,
    distance_hz: float = 2.0,
) -> Dict[str, np.ndarray]:
    """Find peaks in a frequency-domain spectrum.

    Parameters
    ----------
    freqs : 1-D array of frequency bins (Hz).
    amplitudes : 1-D array of spectral amplitudes (same length as *freqs*).
    prominence : Minimum peak prominence as a *fraction of the global maximum*.
        E.g. 0.02 means peaks must be at least 2 % of the max amplitude.
    distance_hz : Minimum separation between peaks in Hz.

    Returns
    -------
    dict with keys:
      ``indices``     - integer indices into *freqs* / *amplitudes*
      ``frequencies`` - Hz values of detected peaks
      ``amplitudes``  - amplitude values at detected peaks
    """
    from scipy.signal import find_peaks as _find_peaks

    if len(freqs) == 0 or len(amplitudes) == 0:
        empty = np.array([], dtype=int)
        return {"indices": empty, "frequencies": np.array([]), "amplitudes": np.array([])}

    freq_resolution = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
    distance_samples = max(1, int(distance_hz / freq_resolution))
    abs_prominence = prominence * float(np.max(amplitudes))

    indices, _ = _find_peaks(
        amplitudes,
        prominence=abs_prominence,
        distance=distance_samples,
    )

    return {
        "indices": indices,
        "frequencies": freqs[indices],
        "amplitudes": amplitudes[indices],
    }


def find_dominant_frequency(
    freqs: np.ndarray,
    amplitudes: np.ndarray,
    freq_min: float = 10.0,
    freq_max: float = 60.0,
) -> Optional[float]:
    """Return the frequency (Hz) of the largest spectral component in [freq_min, freq_max].

    Returns ``None`` if no bins fall in the requested range.
    """
    mask = (freqs >= freq_min) & (freqs <= freq_max)
    if not np.any(mask):
        return None
    idx_in_range = np.where(mask)[0]
    best = idx_in_range[np.argmax(amplitudes[mask])]
    return float(freqs[best])


def find_harmonics(
    freqs: np.ndarray,
    amplitudes: np.ndarray,
    f0: float,
    n_harmonics: int = 5,
    tolerance_hz: float = 2.0,
    prominence: float = 0.02,
    distance_hz: float = 2.0,
) -> List[Dict]:
    """Identify harmonics of *f0* in the spectrum.

    For each k in 1 … *n_harmonics* the expected frequency k*f0 is checked.
    If a detected peak lies within *tolerance_hz* it is labelled as found;
    otherwise the amplitude at k*f0 is interpolated from the spectrum.

    Parameters
    ----------
    freqs, amplitudes : spectrum arrays (from Welch or FFT).
    f0 : fundamental frequency in Hz.
    n_harmonics : number of harmonics to check (k=1 is the fundamental itself).
    tolerance_hz : ± window around k*f0 to search for a matching peak.
    prominence, distance_hz : forwarded to :func:`find_spectral_peaks`.

    Returns
    -------
    List of dicts, one per harmonic:
      ``harmonic_n``   - integer k
      ``expected_hz``  - k * f0
      ``actual_hz``    - frequency of the matched peak (or k*f0 if not found)
      ``amplitude``    - spectral amplitude at *actual_hz*
      ``peak_found``   - True if a detected peak matched within tolerance
    """
    peaks = find_spectral_peaks(freqs, amplitudes, prominence=prominence, distance_hz=distance_hz)
    peak_freqs = peaks["frequencies"]
    peak_amps = peaks["amplitudes"]

    results = []
    for k in range(1, n_harmonics + 1):
        expected = k * f0
        # Search for nearest detected peak within tolerance
        if len(peak_freqs) > 0:
            diffs = np.abs(peak_freqs - expected)
            nearest_idx = int(np.argmin(diffs))
            if diffs[nearest_idx] <= tolerance_hz:
                actual_hz = float(peak_freqs[nearest_idx])
                amp = float(peak_amps[nearest_idx])
                found = True
            else:
                actual_hz = expected
                amp = float(np.interp(expected, freqs, amplitudes))
                found = False
        else:
            actual_hz = expected
            amp = float(np.interp(expected, freqs, amplitudes))
            found = False

        results.append({
            "harmonic_n": k,
            "expected_hz": round(expected, 3),
            "actual_hz": round(actual_hz, 3),
            "amplitude": amp,
            "peak_found": found,
        })

    return results
