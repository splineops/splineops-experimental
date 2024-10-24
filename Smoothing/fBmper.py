"""
fBmper.py

Fractional (pseudo-)Brownian motion generator.

References:
    [1] M. Unser and T. Blu, "Self-Similarity: Part I -- Splines and Operators",
        IEEE Trans. Signal Processing, vol. 55, no. 4, pp. 1352-1363, April 2007.
    [2] T. Blu and M. Unser, "Self-Similarity: Part II -- Optimal Estimation of Fractal Processes",
        IEEE Trans. Signal Processing, vol. 55, no. 4, pp. 1364-1378, April 2007.
    [3] M. Unser and T. Blu, "Fractional Splines and Wavelets," SIAM Review,
        vol. 42, no. 1, pp. 43-67, March 2000.

Author: Assistant, based on original code by Dr. Thierry Blu.

This software can be downloaded at <http://bigwww.epfl.ch/>.
"""

import numpy as np
from fractsplineautocorr import fractsplineautocorr

def fBmper(epsH, H, m, N):
    """
    Fractional (pseudo-)Brownian motion generator.

    Parameters:
    epsH (float): Scaling parameter (variance parameter).
    H (float): Hurst parameter (0 < H < 1).
    m (int): Upsampling factor.
    N (int): Number of samples.

    Returns:
    t (numpy array): Time vector.
    y (numpy array): Generated fractional Brownian motion samples.

    References:
        See above.
    """
    # Generate random Fourier coefficients
    Y = np.fft.fft(np.random.randn(m * N))
    Y = Y[1:]

    omega = np.arange(1, m * N) * 2 * np.pi / (m * N)

    # Compute the modified Fourier coefficients
    Y = (m ** (-H) * epsH * Y /
         np.abs(2 * np.sin(omega / 2)) ** (H + 0.5) *
         np.sqrt(fractsplineautocorr(H - 0.5, omega / (2 * np.pi))))

    # Enforce real-valued signal
    Y = np.concatenate(([-np.real(np.sum(Y))], Y))

    # Inverse FFT to get time domain signal
    y = np.real(np.fft.ifft(Y))
    t = np.arange(0, N, 1 / m)
    return t, y
