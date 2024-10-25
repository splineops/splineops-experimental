import numpy as np
import matplotlib.pyplot as plt
from scipy.datasets import ascent

# Import the resize_image function from your package
from resize import resize_image

# Load the ascent image as a NumPy array
input_img = ascent()  # This is a 512x512 grayscale image with values from 0 to 255

# Convert the input image to float64 for processing, but keep the original for plotting
input_img_float = input_img.astype(np.float64)

input_image_normalized = (input_img / 255.0).astype(np.float64)

# Define the shrink factor
shrink_factor = 0.2  # Shrink the image to 20% of its original size

# Shrink the image using the resize_image function with inversable=True
shrunken_img = resize_image(
    input_img_normalized=input_image_normalized,
    zoom_factors=(shrink_factor, shrink_factor),
    method='Least-Squares',
    interpolation='Linear',
)

# Now expand the shrunken image back to the original size
expanded_img = resize_image(
    input_img_normalized=shrunken_img,
    output_size=input_img.shape,  # Use the original image size
    method='Least-Squares',
    interpolation='Linear',
)

# Convert images back to uint8 for display
input_img_display = input_img  # Original image, already in uint8
shrunken_img_display = np.clip(shrunken_img * 255, 0, 255).astype(np.uint8)
expanded_img_display = np.clip(expanded_img * 255, 0, 255).astype(np.uint8)

# Compute the difference between the original and re-expanded images
difference_img = np.abs(input_image_normalized - expanded_img)
difference_img_display = np.clip(difference_img * 255 * 5, 0, 255).astype(np.uint8)  # Enhanced for visibility

# Compute SNR and MSE
def compute_snr(original, processed):
    # Compute the signal-to-noise ratio
    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - processed) ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def compute_mse(original, processed):
    # Compute the mean squared error
    mse = np.mean((original - processed) ** 2)
    return mse

snr = compute_snr(input_image_normalized, expanded_img)
mse = compute_mse(input_image_normalized, expanded_img)
print(f"SNR: {snr:.2f} dB")
print(f"MSE: {mse:.6f}")

# Plot the images
plt.figure(figsize=(12, 8))

# Original image
plt.subplot(2, 2, 1)
plt.imshow(input_img_display, cmap='gray', vmin=0, vmax=255)
plt.title('Original Image')
plt.axis('off')

# Shrunken image
plt.subplot(2, 2, 2)
plt.imshow(shrunken_img_display, cmap='gray', vmin=0, vmax=255)
plt.title(f'Shrunken Image (Factor {shrink_factor})')
plt.axis('off')

# Re-expanded image
plt.subplot(2, 2, 3)
plt.imshow(expanded_img_display, cmap='gray', vmin=0, vmax=255)
plt.title('Re-expanded Image')
plt.axis('off')

# Difference image
plt.subplot(2, 2, 4)
plt.imshow(difference_img_display, cmap='gray', vmin=0, vmax=255)
plt.title('Difference Image (Enhanced)')
plt.axis('off')

plt.tight_layout()
plt.show()
