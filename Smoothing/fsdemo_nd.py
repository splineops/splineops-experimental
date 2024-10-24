import numpy as np
import matplotlib.pyplot as plt
from smoothspline_nd import smoothspline_nd
from scipy.ndimage import gaussian_filter
from skimage import data

# Use a real image for better assessment
def create_example_image():
    """
    Loads a real grayscale image.
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

# Parameters
lambda_ = 0.01  # Adjusted lambda_
gamma = 2.0     # Adjusted gamma
snr_db = 10.0   # Desired SNR in dB

# Generate example image
img = create_example_image()
# Add noise
noisy_img = add_noise(img, snr_db)

# Apply smoothing spline
smoothed_img = smoothspline_nd(noisy_img, lambda_, gamma)

# Calculate SNR improvement
noise = noisy_img - img
MSE_noisy = np.mean(noise ** 2)
noise_smoothed = smoothed_img - img
MSE_smoothed = np.mean(noise_smoothed ** 2)

SNR_noisy = 10 * np.log10(np.mean(img ** 2) / MSE_noisy)
SNR_smoothed = 10 * np.log10(np.mean(img ** 2) / MSE_smoothed)
SNR_improvement = SNR_smoothed - SNR_noisy

print(f"SNR of noisy image: {SNR_noisy:.2f} dB")
print(f"SNR after smoothing: {SNR_smoothed:.2f} dB")
print(f"SNR improvement: {SNR_improvement:.2f} dB")

# Visualization
plt.figure(figsize=(12, 8))
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(noisy_img, cmap='gray')
plt.title(f'Noisy Image (SNR={SNR_noisy:.2f} dB)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(smoothed_img, cmap='gray')
plt.title(f'Smoothed Image (SNR={SNR_smoothed:.2f} dB)')
plt.axis('off')

plt.tight_layout()
plt.show()
