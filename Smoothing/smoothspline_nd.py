import numpy as np
from scipy.fft import fftn, ifftn

def smoothspline_nd(data, lambda_, gamma):
    """
    Applies multi-dimensional fractional smoothing spline to the input data.

    Parameters:
    data (ndarray): Multi-dimensional input data (e.g., image, volume).
    lambda_ (float): Regularization parameter.
    gamma (float): Order of the spline operator (gamma = H + 0.5).

    Returns:
    data_smooth (ndarray): Smoothed data.
    """
    data = np.asarray(data)
    dims = data.shape

    # Compute the frequency grids for each dimension
    freq_grids = np.meshgrid(*[np.fft.fftfreq(n) for n in dims], indexing='ij')
    omega_squared = np.zeros(dims)
    for grid in freq_grids:
        omega_squared += (2 * np.pi * grid) ** 2
    # Removed fftshift from omega_squared
    # omega_squared = fftshift(omega_squared)

    # Compute the Butterworth-like filter in Fourier domain
    H = 1 / (1 + lambda_ * (omega_squared) ** gamma)

    # Apply the filter
    data_fft = fftn(data)
    data_smooth_fft = H * data_fft
    data_smooth = np.real(ifftn(data_smooth_fft))

    return data_smooth
