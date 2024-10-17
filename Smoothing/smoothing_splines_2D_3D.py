import numpy as np
from scipy.fft import fftn, ifftn, fftfreq
import matplotlib.pyplot as plt

def butterworth_filter_2d(data, cutoff_freq, order):
    """
    Apply a 2D Butterworth low-pass filter to a 2D image.

    Parameters:
    data (np.ndarray): 2D input data (e.g., an image).
    cutoff_freq (float): Cutoff frequency of the filter.
    order (int): Order of the Butterworth filter (controls sharpness of cutoff).

    Returns:
    np.ndarray: Smoothed data after applying the Butterworth filter.
    """
    # Get the shape of the data
    rows, cols = data.shape
    
    # Create 2D frequency grid
    u = fftfreq(rows).reshape(-1, 1)
    v = fftfreq(cols).reshape(1, -1)
    
    # Compute the distance from the origin in the frequency domain
    D_uv = np.sqrt(u**2 + v**2)
    
    # Create the Butterworth filter in the frequency domain
    H_uv = 1 / (1 + (D_uv / cutoff_freq)**(2 * order))
    
    # Apply the filter: FFT -> multiply -> IFFT
    data_fft = fftn(data)
    filtered_fft = data_fft * H_uv
    filtered_data = np.real(ifftn(filtered_fft))
    
    return filtered_data

def butterworth_filter_3d(data, cutoff_freq, order):
    """
    Apply a 3D Butterworth low-pass filter to a 3D volume.

    Parameters:
    data (np.ndarray): 3D input data (e.g., a volumetric image).
    cutoff_freq (float): Cutoff frequency of the filter.
    order (int): Order of the Butterworth filter (controls sharpness of cutoff).

    Returns:
    np.ndarray: Smoothed data after applying the Butterworth filter.
    """
    # Get the shape of the data
    depth, rows, cols = data.shape
    
    # Create 3D frequency grid
    u = fftfreq(depth).reshape(-1, 1, 1)
    v = fftfreq(rows).reshape(1, -1, 1)
    w = fftfreq(cols).reshape(1, 1, -1)
    
    # Compute the distance from the origin in the frequency domain
    D_uvw = np.sqrt(u**2 + v**2 + w**2)
    
    # Create the Butterworth filter in the frequency domain
    H_uvw = 1 / (1 + (D_uvw / cutoff_freq)**(2 * order))
    
    # Apply the filter: FFT -> multiply -> IFFT
    data_fft = fftn(data)
    filtered_fft = data_fft * H_uvw
    filtered_data = np.real(ifftn(filtered_fft))
    
    return filtered_data

# Example usage for 2D data (image)
def demo_2d_butterworth():
    # Create a 2D noisy image (for demo purposes)
    x = np.linspace(0, 1, 256)
    y = np.linspace(0, 1, 256)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(4 * np.pi * X) + np.sin(4 * np.pi * Y) + 0.3 * np.random.randn(256, 256)
    
    # Apply Butterworth filter
    cutoff_freq = 0.1
    order = 2
    Z_smooth = butterworth_filter_2d(Z, cutoff_freq, order)
    
    # Plot the original and filtered images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Noisy Image')
    plt.imshow(Z, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title('Filtered Image (Butterworth)')
    plt.imshow(Z_smooth, cmap='gray')
    plt.show()

# Example usage for 3D data (volume)
def demo_3d_butterworth():
    # Create a 3D noisy volume (for demo purposes)
    x = np.linspace(0, 1, 64)
    y = np.linspace(0, 1, 64)
    z = np.linspace(0, 1, 64)
    X, Y, Z = np.meshgrid(x, y, z)
    V = np.sin(4 * np.pi * X) + np.sin(4 * np.pi * Y) + np.sin(4 * np.pi * Z) + 0.3 * np.random.randn(64, 64, 64)
    
    # Apply Butterworth filter
    cutoff_freq = 0.1
    order = 2
    V_smooth = butterworth_filter_3d(V, cutoff_freq, order)
    
    # Visualize one slice of the original and filtered volumes
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Noisy Volume Slice')
    plt.imshow(V[32, :, :], cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title('Filtered Volume Slice (Butterworth)')
    plt.imshow(V_smooth[32, :, :], cmap='gray')
    plt.show()

# Run the 2D demo
demo_2d_butterworth()

# Run the 3D demo
demo_3d_butterworth()
