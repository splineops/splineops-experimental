"""
fsdemo_nd_cameraman.py

Smoothing spline demo using Butterworth filter approximation on the cameraman image.

Author: Assistant.

This software can be downloaded at <http://bigwww.epfl.ch/>.
"""

import numpy as np
import matplotlib.pyplot as plt
from smoothspline_nd import smoothspline_nd
from skimage import data

def create_camera_image():
    """
    Loads a real grayscale image (cameraman).
    """
    img = data.camera().astype(np.float64)
    img /= 255.0  # Normalize to [0, 1]
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

def demo_cameraman_image():
    # Parameters
    lambda_ = 0.1  # Regularization parameter
    gamma = 2.0     # Order of the spline operator
    snr_db = 10.0   # Desired SNR in dB

    # Load cameraman image
    img_camera = create_camera_image()
    noisy_img_camera = add_noise(img_camera, snr_db)
    smoothed_img_camera = smoothspline_nd(noisy_img_camera, lambda_, gamma)

    # Compute SNRs
    snr_noisy_camera = compute_snr(img_camera, noisy_img_camera)
    snr_smooth_camera = compute_snr(img_camera, smoothed_img_camera)
    snr_improvement_camera = snr_smooth_camera - snr_noisy_camera

    print("Cameraman Image:")
    print(f"SNR of noisy image: {snr_noisy_camera:.2f} dB")
    print(f"SNR after smoothing: {snr_smooth_camera:.2f} dB")
    print(f"SNR improvement: {snr_improvement_camera:.2f} dB\n")

    # Visualization for Cameraman Image
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img_camera, cmap='gray')
    plt.title('Original Cameraman Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(noisy_img_camera, cmap='gray')
    plt.title(f'Noisy Image (SNR={snr_noisy_camera:.2f} dB)')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(smoothed_img_camera, cmap='gray')
    plt.title(f'Smoothed Image (SNR={snr_smooth_camera:.2f} dB)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run the cameraman image demo
    demo_cameraman_image()
