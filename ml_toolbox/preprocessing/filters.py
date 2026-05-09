"""
Digital filter implementations for signal preprocessing.
"""

import numpy as np
from scipy import signal as _signal
from typing import Optional


class ButterworthLPF:
    """Zero-phase Butterworth low-pass filter.

    Parameters
    ----------
    cutoff_hz : float
        Cutoff frequency in Hz.
    order : int, optional
        Filter order. Default is 4.
    """

    def __init__(self, cutoff_hz: float, order: int = 4):
        if cutoff_hz <= 0:
            raise ValueError(f"cutoff_hz must be positive, got {cutoff_hz}")
        if order < 1:
            raise ValueError(f"order must be >= 1, got {order}")
        self.cutoff_hz = cutoff_hz
        self.order = order

    def apply(self, data: np.ndarray, fs: float) -> np.ndarray:
        """Apply the filter to a signal.

        Uses zero-phase filtering (``scipy.signal.filtfilt``) to avoid phase
        distortion. Works on 1-D arrays and 2-D arrays with shape
        ``(samples, channels)`` — each channel is filtered independently.

        Parameters
        ----------
        data : np.ndarray
            Input signal. Shape ``(n_samples,)`` or ``(n_samples, n_channels)``.
        fs : float
            Sampling rate of *data* in Hz.

        Returns
        -------
        np.ndarray
            Filtered signal with the same shape and dtype as *data*.
        """
        nyq = fs / 2.0
        if self.cutoff_hz >= nyq:
            raise ValueError(
                f"cutoff_hz ({self.cutoff_hz} Hz) must be less than the "
                f"Nyquist frequency ({nyq} Hz) for fs={fs} Hz."
            )
        normalized_cutoff = self.cutoff_hz / nyq
        b, a = _signal.butter(self.order, normalized_cutoff, btype='low')
        return _signal.filtfilt(b, a, data, axis=0).astype(data.dtype)

    def __repr__(self) -> str:
        return f"ButterworthLPF(cutoff_hz={self.cutoff_hz}, order={self.order})"
