"""
smoothspline.py

Fractional smoothing spline.

Returns samples of the smoothing spline for a given input sequence,
sampled at "m" times the rate of input. The input is assumed to be
sampled at integers 0..N-1.

References:
    [1] M. Unser and T. Blu, "Self-Similarity: Part I -- Splines and Operators",
        IEEE Trans. Signal Processing, vol. 55, no. 4, pp. 1352-1363, April 2007.
    [2] T. Blu and M. Unser, "Self-Similarity: Part II -- Optimal Estimation of Fractal Processes",
        IEEE Trans. Signal Processing, vol. 55, no. 4, pp. 1364-1378, April 2007.
    [3] M. Unser and T. Blu, "Fractional Splines and Wavelets," SIAM Review,
        vol. 42, no. 1, pp. 43-67, March 2000.

Author: Assistant, based on code by Dr. Thierry Blu.

This software can be downloaded at <http://bigwww.epfl.ch/>.
"""

import numpy as np
from fractsplineautocorr import fractsplineautocorr

def periodize(x, m):
    """
    Periodizes the input array by concatenating 'm' copies of it.

    Parameters:
    x (numpy array): Input array.
    m (int): Number of times to concatenate.

    Returns:
    xp (numpy array): Periodized array.
    """
    return np.tile(x, m)

def smoothspline(y, lambda_, m, gamma):
    """
    Computes the fractional smoothing spline of an input signal.

    Parameters:
    y (array-like): Input signal.
    lambda_ (float): Regularization parameter.
    m (int): Upsampling factor.
    gamma (float): Order of the spline operator (gamma = H + 0.5).

    Returns:
    t (numpy array): Time vector.
    ys (numpy array): Smoothing spline sequence.

    References:
        See above.
    """
    y = np.asarray(y).flatten()
    N = len(y)

    # Compute the FFT of the input signal
    Y = np.fft.fft(y)
    omega = np.arange(1, N * m) * 2 * np.pi / (N * m)

    # Upsample Y
    Ym = periodize(Y, m)

    # Internal calculations
    sinm2g = np.abs(2 * np.sin(m * omega / 2)) ** (2 * gamma)
    sin2g = np.abs(2 * np.sin(omega / 2)) ** (2 * gamma)

    # Calculate A_gamma(omega)
    alpha = gamma - 1
    Ag = fractsplineautocorr(alpha, np.concatenate(([0], omega / (2 * np.pi))))

    # Calculate A_gamma(m * omega)
    Agm = fractsplineautocorr(alpha, np.concatenate(([0], m * omega / (2 * np.pi))))

    Ag = Ag[1:]
    Agm = Agm[1:]

    # Compute the smoothing spline filter H_m
    Hm = (m ** (-2 * gamma + 1) * (sinm2g / sin2g) * Ag /
          (Agm + lambda_ * sinm2g))
    Hm = np.concatenate(([m], Hm))

    # Generate outputs
    ys = np.real(np.fft.ifft(Hm * Ym))
    t = np.arange(0, N, 1 / m)
    return t, ys
