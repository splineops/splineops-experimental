import numpy as np
from scipy.fft import fftn, ifftn, fftfreq
import matplotlib.pyplot as plt

def butterworth_filter_nd(data, cutoff_freq, order):
    """
    Apply a Butterworth low-pass filter to N-dimensional data (2D, 3D, etc.).

    Parameters:
    data (np.ndarray): N-dimensional input data (e.g., 2D image, 3D volume).
    cutoff_freq (float): Cutoff frequency of the filter.
    order (int): Order of the Butterworth filter (controls sharpness of cutoff).

    Returns:
    np.ndarray: Smoothed data after applying the Butterworth filter.
    """
    # Get the shape of the data
    shape = data.shape
    N = len(shape)  # Number of dimensions
    
    # Create frequency grids for each dimension
    freqs = [fftfreq(s).reshape([-1 if i == dim else 1 for i, s in enumerate(shape)]) for dim, s in enumerate(shape)]
    
    # Compute the distance from the origin in the frequency domain (Euclidean distance in N-dimensions)
    D = np.sqrt(sum(f**2 for f in freqs))
    
    # Create the Butterworth filter in the frequency domain
    H = 1 / (1 + (D / cutoff_freq)**(2 * order))
    
    # Apply the filter: FFT -> multiply -> IFFT
    data_fft = fftn(data)
    filtered_fft = data_fft * H
    filtered_data = np.real(ifftn(filtered_fft))
    
    return filtered_data

def compute_snr(clean_signal, noisy_signal):
    """
    Compute the Signal-to-Noise Ratio (SNR).

    Parameters:
    clean_signal (np.ndarray): Original clean signal.
    noisy_signal (np.ndarray): Noisy signal.

    Returns:
    float: SNR value in decibels (dB).
    """
    signal_power = np.mean(clean_signal ** 2)
    noise_power = np.mean((noisy_signal - clean_signal) ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

# Example usage for 2D data (image)
def demo_2d_butterworth():
    # Create a 2D clean signal (for demo purposes)
    x = np.linspace(0, 1, 256)
    y = np.linspace(0, 1, 256)
    X, Y = np.meshgrid(x, y)
    clean_Z = np.sin(4 * np.pi * X) + np.sin(4 * np.pi * Y)
    
    # Add noise to create a noisy image
    noisy_Z = clean_Z + 0.3 * np.random.randn(256, 256)
    
    # Compute SNR before filtering
    snr_before = compute_snr(clean_Z, noisy_Z)
    
    # Apply Butterworth filter
    cutoff_freq = 0.001
    order = 2
    Z_smooth = butterworth_filter_nd(noisy_Z, cutoff_freq, order)
    
    # Compute SNR after filtering
    snr_after = compute_snr(clean_Z, Z_smooth)
    
    # Plot the original, noisy, and filtered images
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title('Clean Image')
    plt.imshow(clean_Z, cmap='gray')
    
    plt.subplot(1, 3, 2)
    plt.title(f'Noisy Image (SNR: {snr_before:.2f} dB)')
    plt.imshow(noisy_Z, cmap='gray')
    
    plt.subplot(1, 3, 3)
    plt.title(f'Filtered Image (SNR: {snr_after:.2f} dB)')
    plt.imshow(Z_smooth, cmap='gray')
    
    plt.show()

# Example usage for 3D data (volume)
def demo_3d_butterworth():
    # Create a 3D clean volume (for demo purposes)
    x = np.linspace(0, 1, 64)
    y = np.linspace(0, 1, 64)
    z = np.linspace(0, 1, 64)
    X, Y, Z = np.meshgrid(x, y, z)
    clean_V = np.sin(4 * np.pi * X) + np.sin(4 * np.pi * Y) + np.sin(4 * np.pi * Z)
    
    # Add noise to create a noisy volume
    noisy_V = clean_V + 0.3 * np.random.randn(64, 64, 64)
    
    # Compute SNR before filtering
    snr_before = compute_snr(clean_V, noisy_V)
    
    # Apply Butterworth filter
    cutoff_freq = 0.1
    order = 2
    V_smooth = butterworth_filter_nd(noisy_V, cutoff_freq, order)
    
    # Compute SNR after filtering
    snr_after = compute_snr(clean_V, V_smooth)
    
    # Visualize one slice of the original and filtered volumes
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title('Clean Volume Slice')
    plt.imshow(clean_V[32, :, :], cmap='gray')
    
    plt.subplot(1, 3, 2)
    plt.title(f'Noisy Volume Slice (SNR: {snr_before:.2f} dB)')
    plt.imshow(noisy_V[32, :, :], cmap='gray')
    
    plt.subplot(1, 3, 3)
    plt.title(f'Filtered Volume Slice (SNR: {snr_after:.2f} dB)')
    plt.imshow(V_smooth[32, :, :], cmap='gray')
    
    plt.show()

# Run the 2D demo
demo_2d_butterworth()

# Run the 3D demo
demo_3d_butterworth()
