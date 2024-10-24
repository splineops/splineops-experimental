"""
fractional_smoothing_spline.py

This script implements the fractional smoothing spline estimator for 1D signals,
as well as generates fractional Brownian motion (fBm). The code is based on 
the research work by Michael Unser and Thierry Blu, which links fractional 
splines and fractals.

References:
[1] M. Unser, T. Blu, "Fractional Splines and Wavelets," SIAM Review, vol. 42, 
    no. 1, pp. 43-67, March 2000.

[2] M. Unser, T. Blu, "Self-Similarity: Part I—Splines and Operators," 
    IEEE Transactions on Signal Processing, 2007.

[3] T. Blu, M. Unser, "Self-Similarity: Part II—Optimal Estimation of Fractal 
    Processes," IEEE Transactions on Signal Processing, 2007.

This implementation includes:
- Fractional smoothing spline estimator (using FFT)
- Fractional Brownian motion generator
- Autocorrelation function for fractional splines

The original MATLAB code was developed by the Biomedical Imaging Group (BIG) 
at EPFL and provided in the "Fractional Splines and Fractals" package.

© 2024 EPFL
"""

import numpy as np
from scipy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt
from fbm import FBM

# 1. Generate fractional Brownian motion (fBm)
def generate_fbm(n, hurst, length=1):
    """
    Generates a fractional Brownian motion signal with specified parameters.

    Parameters:
    n (int): Number of points in the signal.
    hurst (float): Hurst parameter, which controls the roughness of the signal.
    length (float): Length of the signal.

    Returns:
    np.ndarray: Generated fBm signal.
    """
    fbm_instance = FBM(n=n, hurst=hurst, length=length)
    return fbm_instance.fbm()

# 2. Compute autocorrelation
def autocorrelation(signal):
    """
    Computes the autocorrelation of a signal.

    Parameters:
    signal (np.ndarray): Input signal.

    Returns:
    np.ndarray: Autocorrelation of the input signal.
    """
    result = np.correlate(signal, signal, mode='full')
    return result[result.size // 2:] / np.max(result)

# 3. Apply fractional smoothing spline using FFT
def fractional_smoothing_spline(data, alpha=1.5, lam=0.01):
    """
    Applies a fractional smoothing spline filter to the input data using FFT.

    Parameters:
    data (np.ndarray): Input data to be smoothed.
    alpha (float): Fractional derivative order. Higher values lead to more smoothing.
    lam (float): Regularization parameter. Higher values lead to more smoothing.

    Returns:
    np.ndarray: Smoothed data.
    """
    N = len(data)
    freq = fftfreq(N)
    omega = 2 * np.pi * freq
    
    # Fourier transform of the data
    data_fft = fft(data)
    
    # Fractional smoothing filter
    H = 1 / (1 + lam * np.abs(omega) ** (2 * alpha))
    
    # Apply the filter in the frequency domain
    smoothed_fft = data_fft * H
    
    # Inverse FFT to get the smoothed signal
    smoothed_signal = np.real(ifft(smoothed_fft))
    
    return smoothed_signal

# 4. Demo of the entire process with three different smoothing splines
def demo():
    """
    Demonstrates the use of fractional smoothing splines with three different parameter sets,
    showing how varying alpha and lambda affect the smoothness of the result.
    """
    # Generate noisy signal
    x = np.linspace(0, 1, 1024)
    y = np.sin(2 * np.pi * x) + 0.2 * np.random.randn(1024)

    # Apply smoothing splines with different parameters
    y_smooth_1 = fractional_smoothing_spline(y, alpha=1.5, lam=0.01)
    y_smooth_2 = fractional_smoothing_spline(y, alpha=2.0, lam=0.1)
    y_smooth_3 = fractional_smoothing_spline(y, alpha=2.0, lam=1.0)

    # Plot
    plt.plot(x, y, label='Noisy Data', color='gray')
    plt.plot(x, y_smooth_1, label='Smooth (α=1.5, λ=0.01)')
    plt.plot(x, y_smooth_2, label='Smoother (α=2.0, λ=0.1)')
    plt.plot(x, y_smooth_3, label='Very Smooth (α=2.0, λ=1.0)')
    plt.legend()
    plt.title("Fractional Smoothing Splines with Different Parameters")
    plt.show()

# Run the demo
if __name__ == "__main__":
    demo()
