"""
fsdemo_nd_sinusoid.py

Smoothing spline demo using Butterworth filter approximation on sinusoid images and volumes.

Author: Assistant.

This software can be downloaded at <http://bigwww.epfl.ch/>.
"""

import numpy as np
import matplotlib.pyplot as plt
from smoothspline_nd import smoothspline_nd

def create_sinusoid_image(size=(256, 256)):
    """
    Creates a synthetic 2D sinusoid image.
    """
    x = np.linspace(0, 1, size[1])
    y = np.linspace(0, 1, size[0])
    X, Y = np.meshgrid(x, y)
    img = np.sin(8 * np.pi * X) + np.sin(8 * np.pi * Y)
    img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
    return img

def add_noise(img, snr_db):
    """
    Adds Gaussian noise to the image based on the desired SNR in dB.
    """
    signal_power = np.mean(img ** 2)
    sigma = np.sqrt(signal_power / (10 ** (snr_db / 10)))
    noise = np.random.randn(*img.shape) * sigma
    noisy_img = img + noise
    return noisy_img

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

def demo_sinusoid_image():
    # Desired cutoff frequency
    cutoff_freq = 0.01  # Adjusted cutoff frequency
    gamma = 3.0        # Order of the spline operator

    # Compute lambda_ based on cutoff frequency
    lambda_ = (1 / (2 * np.pi * cutoff_freq)) ** (2 * gamma)

    snr_db = 10.0   # Desired SNR in dB

    # Sinusoid image
    img_sinusoid = create_sinusoid_image()
    noisy_img_sinusoid = add_noise(img_sinusoid, snr_db)
    smoothed_img_sinusoid = smoothspline_nd(noisy_img_sinusoid, lambda_, gamma)

    # Compute SNRs
    snr_noisy_sinusoid = compute_snr(img_sinusoid, noisy_img_sinusoid)
    snr_smooth_sinusoid = compute_snr(img_sinusoid, smoothed_img_sinusoid)
    snr_improvement_sinusoid = snr_smooth_sinusoid - snr_noisy_sinusoid

    print("Sinusoid Image:")
    print(f"SNR of noisy image: {snr_noisy_sinusoid:.2f} dB")
    print(f"SNR after smoothing: {snr_smooth_sinusoid:.2f} dB")
    print(f"SNR improvement: {snr_improvement_sinusoid:.2f} dB\n")

    # Visualization for Sinusoid Image
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img_sinusoid, cmap='gray')
    plt.title('Original Sinusoid Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(noisy_img_sinusoid, cmap='gray')
    plt.title(f'Noisy Image (SNR={snr_noisy_sinusoid:.2f} dB)')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(smoothed_img_sinusoid, cmap='gray')
    plt.title(f'Smoothed Image (SNR={snr_smooth_sinusoid:.2f} dB)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def demo_3d_sinusoid():
    # Desired cutoff frequency
    cutoff_freq = 0.1  # Adjusted cutoff frequency
    gamma = 2.0        # Order of the spline operator

    # Compute lambda_ based on cutoff frequency
    lambda_ = (1 / (2 * np.pi * cutoff_freq)) ** (2 * gamma)

    snr_db = 10.0   # Desired SNR in dB

    # Create a 3D clean volume (sinusoid)
    x = np.linspace(0, 1, 64)
    y = np.linspace(0, 1, 64)
    z = np.linspace(0, 1, 64)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    clean_volume = np.sin(8 * np.pi * X) + np.sin(8 * np.pi * Y) + np.sin(8 * np.pi * Z)
    clean_volume = (clean_volume - clean_volume.min()) / (clean_volume.max() - clean_volume.min())  # Normalize to [0, 1]

    # Add noise
    signal_power = np.mean(clean_volume ** 2)
    sigma = np.sqrt(signal_power / (10 ** (snr_db / 10)))
    noise = np.random.randn(*clean_volume.shape) * sigma
    noisy_volume = clean_volume + noise

    # Apply smoothing spline
    smoothed_volume = smoothspline_nd(noisy_volume, lambda_, gamma)

    # Compute SNRs
    snr_noisy = compute_snr(clean_volume, noisy_volume)
    snr_smoothed = compute_snr(clean_volume, smoothed_volume)
    snr_improvement = snr_smoothed - snr_noisy

    print("3D Sinusoid Volume:")
    print(f"SNR of noisy volume: {snr_noisy:.2f} dB")
    print(f"SNR after smoothing: {snr_smoothed:.2f} dB")
    print(f"SNR improvement: {snr_improvement:.2f} dB\n")

    # Visualize one slice of the volume (middle slice)
    slice_index = clean_volume.shape[2] // 2

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(clean_volume[:, :, slice_index], cmap='gray')
    plt.title('Clean Volume Slice')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(noisy_volume[:, :, slice_index], cmap='gray')
    plt.title(f'Noisy Slice (SNR={snr_noisy:.2f} dB)')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(smoothed_volume[:, :, slice_index], cmap='gray')
    plt.title(f'Smoothed Slice (SNR={snr_smoothed:.2f} dB)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run the sinusoid image demo
    demo_sinusoid_image()
    # Run the 3D sinusoid demo
    demo_3d_sinusoid()
