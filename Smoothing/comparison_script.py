import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn, fftfreq

def smoothspline_nd(data, lambda_, gamma):
    data = np.asarray(data)
    dims = data.shape

    # Compute the frequency grids for each dimension
    freq_grids = np.meshgrid(*[fftfreq(n) for n in dims], indexing='ij')
    omega_squared = np.zeros(dims)
    for grid in freq_grids:
        omega_squared += (2 * np.pi * grid) ** 2

    # Compute the Butterworth-like filter in Fourier domain
    H = 1 / (1 + lambda_ * omega_squared ** gamma)

    # Apply the filter
    data_fft = fftn(data)
    data_smooth_fft = H * data_fft
    data_smooth = np.real(ifftn(data_smooth_fft))

    return data_smooth

def butterworth_filter_nd(data, cutoff_freq, order):
    shape = data.shape
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
    signal_power = np.mean(clean_signal ** 2)
    noise_power = np.mean((noisy_signal - clean_signal) ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

# Create a 2D sinusoid image
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

# Add noise to the image
def add_noise(img, snr_db):
    signal_power = np.mean(img ** 2)
    sigma = np.sqrt(signal_power / (10 ** (snr_db / 10)))
    noise = np.random.randn(*img.shape) * sigma
    noisy_img = img + noise
    return noisy_img

# Parameters
snr_db = 10.0  # Desired SNR in dB
original_img = create_sinusoid_image()  # Use the sinusoid image
noisy_img = add_noise(original_img, snr_db)

# Compute SNR before filtering
snr_noisy = compute_snr(original_img, noisy_img)

# Parameters for filters
gamma = 3.0  # For smoothspline_nd
n = gamma    # Order of Butterworth filter

# Relation between lambda_ and cutoff_freq
lambda_ = 0.1  # Choose lambda_
cutoff_freq = 1 / (2 * np.pi * lambda_ ** (1 / (2 * n)))

# Apply smoothspline_nd
smoothed_img_spline = smoothspline_nd(noisy_img, lambda_, gamma)

# Apply butterworth_filter_nd
smoothed_img_butterworth = butterworth_filter_nd(noisy_img, cutoff_freq, int(n))

# Compute SNR after filtering
snr_spline = compute_snr(original_img, smoothed_img_spline)
snr_butterworth = compute_snr(original_img, smoothed_img_butterworth)

# Print SNR improvements
print(f"SNR of noisy image: {snr_noisy:.2f} dB")
print(f"SNR after smoothing spline: {snr_spline:.2f} dB")
print(f"SNR improvement (spline): {snr_spline - snr_noisy:.2f} dB")
print(f"SNR after Butterworth filter: {snr_butterworth:.2f} dB")
print(f"SNR improvement (Butterworth): {snr_butterworth - snr_noisy:.2f} dB")

# Plot the results
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(noisy_img, cmap='gray')
plt.title(f'Noisy Image (SNR: {snr_noisy:.2f} dB)')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(smoothed_img_spline, cmap='gray')
plt.title(f'Smoothed Image (Spline, SNR: {snr_spline:.2f} dB)')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(smoothed_img_butterworth, cmap='gray')
plt.title(f'Smoothed Image (Butterworth, SNR: {snr_butterworth:.2f} dB)')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(original_img, cmap='gray')
plt.title('Original Sinusoid Image')
plt.axis('off')

plt.tight_layout()
plt.show()
