import numpy as np
from fractsplineautocorr import fractsplineautocorr

def fBmper(epsH, H, m, N):
    """
    Fractional (pseudo-)Brownian motion generator.

    Parameters:
    epsH (float): Scaling parameter.
    H (float): Hurst parameter (0 < H < 1).
    m (int): Upsampling factor.
    N (int): Number of samples.

    Returns:
    t (numpy array): Time vector.
    y (numpy array): Generated fractional Brownian motion samples.
    """
    # Generate random Fourier coefficients
    Y = np.fft.fft(np.random.randn(m * N))
    Y = Y[1:]

    omega = np.arange(1, m * N) * 2 * np.pi / (m * N)

    # Compute the modified Fourier coefficients
    Y = (m ** (-H) * epsH * Y /
         np.abs(2 * np.sin(omega / 2)) ** (H + 0.5) *
         np.sqrt(fractsplineautocorr(H - 0.5, omega / (2 * np.pi))))

    Y = np.concatenate(([-np.real(np.sum(Y))], Y))

    y = np.real(np.fft.ifft(Y))
    t = np.arange(0, N, 1 / m)
    return t, y
