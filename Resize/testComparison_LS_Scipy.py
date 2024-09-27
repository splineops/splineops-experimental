import os
from scipy.datasets import ascent
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import imageio.v2 as imageio
from Resize import resize_image
import time

def create_square_image():
    img = np.zeros((10, 10))  # Create a 3x3 black image
    img[3:7, 3:7] = 255.0  # Add a white dot in the center
    return img

# Function to convert the scipy image to numpy array
def load_ascent_image():
    img = ascent()  # Load the ascent image from scipy
    img_resized = zoom(img, (256 / img.shape[0], 256 / img.shape[1]), order=3)  # Resize to 256x256
    return img_resized  # Keep the image values in the range [0, 255]

# Function to load images from the Samples folder
def load_image(filename):
    img_path = os.path.join(os.path.dirname(__file__), 'Samples', filename)
    img = imageio.imread(img_path)
    # Convert to grayscale by averaging the color channels
    if len(img.shape) == 3 and img.shape[2] == 4:  # If the image has 4 channels (RGBA)
        img = np.mean(img[:, :, :3], axis=2)  # Ignore the alpha channel and average RGB channels
    img_resized = zoom(img, (256 / img.shape[0], 256 / img.shape[1]), order=3)  # Resize to 256x256

     # Convert the image values to uint8 if necessary
    if img_resized.dtype != np.uint8:
        img_resized = np.clip(img_resized, 0, 255).astype(np.uint8)
    
    # Save the resized image
    output_filename = os.path.splitext(filename)[0] + '_256x256' + os.path.splitext(filename)[1]
    output_path = os.path.join(os.path.dirname(__file__), 'Samples', output_filename)
    imageio.imwrite(output_path, img_resized)

    return img_resized  # Keep the image values in the range [0, 255]

def compute_snr(original, processed):
    # Compute the signal-to-noise ratio
    signal_power = 255.0 ** 2
    noise_power = np.mean((original - processed) ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def compute_mse(original, processed):
    # Compute the mean squared error
    mse = np.mean((original - processed) ** 2)
    return mse

def resize_and_compute_snr(input_image_normalized, method, interpolation, zoom_factor):
    # Normalize the image to [0, 1] for the resize function
    
    # Perform shrinking
    shrunken_image = resize_image(
        input_img_normalized=input_image_normalized,
        zoom_factors=(zoom_factor, zoom_factor),
        method=method,
        interpolation=interpolation
    )
    
    # Perform expansion back to the original size
    expanded_image = resize_image(
        input_img_normalized=shrunken_image,
        output_size=input_image_normalized.shape,
        method=method,
        interpolation=interpolation
    )

    # Compute SNR and MSE
    snr = compute_snr(input_image_normalized, expanded_image)
    mse = compute_mse(input_image_normalized, expanded_image)

    # Convert images back to range [0, 255] for display
    shrunken_image_display = np.clip(shrunken_image * 255.0, 0, 255)
    expanded_image_display = np.clip(expanded_image * 255.0, 0, 255)
    
    return shrunken_image_display, expanded_image_display, snr, mse

def resize_with_scipy_zoom(input_image_normalized, zoom_factor, interpolation):
    # Set order based on interpolation type
    if interpolation == "Linear":
        order = 1
    elif interpolation == "Quadratic":
        order = 2
    else:  # Cubic
        order = 3

    resized_image = zoom(input_image_normalized, (zoom_factor, zoom_factor), order=order)
    reverse_zoom_factor = 1.0 / zoom_factor
    reverse_resized_image = zoom(resized_image, (reverse_zoom_factor, reverse_zoom_factor), order=order)

    # Resize the reverse output image to match the original dimensions
    zoom_factors = (input_image_normalized.shape[0] / reverse_resized_image.shape[0],
                    input_image_normalized.shape[1] / reverse_resized_image.shape[1])
    resized_reverse_output_image = zoom(reverse_resized_image, zoom_factors, order=order)

    # Compute SNR and MSE
    snr = compute_snr(input_image_normalized, resized_reverse_output_image)
    mse = compute_mse(input_image_normalized, resized_reverse_output_image)

    # Convert images to range [0, 255] for display
    resized_image_display = np.clip(resized_image * 255.0, 0, 255)
    resized_reverse_output_image_display = np.clip(resized_reverse_output_image * 255.0, 0, 255)

    return resized_image_display, resized_reverse_output_image_display, snr, mse

def create_black_background(image, original_shape):
    # Create a black background with the original image dimensions
    black_background = np.zeros(original_shape)
    # Place the image in the top-left corner
    black_background[:image.shape[0], :image.shape[1]] = image
    return black_background

def main():
    # Load the ascent image
    input_image = load_ascent_image()  # Use this to load the ascent image

    # Load chirp.tif image
    #input_image = load_image('collagen-mip.tif')

    # Load collagen-mip.tif image
    #input_image = load_image('chirp.tif')

    #input_image = load_image('headMRI.tif')

    #input_image = create_square_image()  # Use this to load the ascent image

    # Normalize the input image to [0, 1]
    input_image_normalized = (input_image / 255.0).astype(np.float64)

    # Define the zoom factor
    zoom_factor = 1/3.14

    # Define method and interpolation type
    method = "Least-Squares"
    interpolation_type = "Linear"

    # Measure time and process images with least squares method
    start_time = time.time()
    ls_output_image, ls_resized_reverse_output_image, ls_snr, ls_mse = resize_and_compute_snr(input_image_normalized, method, interpolation_type, zoom_factor)
    ls_time = time.time() - start_time

    # Measure time and process images with scipy zoom
    start_time = time.time()
    scipy_output_image, scipy_resized_reverse_output_image, scipy_snr, scipy_mse = resize_with_scipy_zoom(input_image_normalized, zoom_factor, interpolation_type)
    scipy_time = time.time() - start_time

    # Convert the original image for display
    input_image_display = np.clip(input_image_normalized * 255.0, 0, 255)

    # Create black backgrounds if zoom factor is less than 1.0
    if zoom_factor < 1.0:
        ls_output_image_bg = create_black_background(ls_output_image, input_image_display.shape)
        scipy_output_image_bg = create_black_background(scipy_output_image, input_image_display.shape)
    else:
        ls_output_image_bg = ls_output_image
        scipy_output_image_bg = scipy_output_image

    # Compute the difference image for the reversed images
    enhancement_factor = 1
    ls_difference_image = np.abs(input_image_normalized - ls_resized_reverse_output_image / 255.0) * enhancement_factor
    scipy_difference_image = np.abs(input_image_normalized - scipy_resized_reverse_output_image / 255.0) * enhancement_factor
    ls_diff_max = np.max(ls_difference_image)
    ls_diff_mean = np.mean(ls_difference_image)
    scipy_diff_max = np.max(scipy_difference_image)
    scipy_diff_mean = np.mean(scipy_difference_image)

    ls_difference_image_enhanced = ls_difference_image * enhancement_factor
    scipy_difference_image_enhanced = scipy_difference_image * enhancement_factor

    ls_difference_image_enhanced = np.clip(ls_difference_image_enhanced, 0, 1)
    scipy_difference_image_enhanced = np.clip(scipy_difference_image_enhanced, 0, 1)

    # Print statistics to the terminal
    print(f"Least-Squares Method:")
    print(f"  SNR: {ls_snr:.2f} dB")
    print(f"  MSE: {ls_mse:.2e}")
    print(f"  Max Difference: {ls_diff_max:.2e}")
    print(f"  Mean Difference: {ls_diff_mean:.2e}")
    print(f"  Time: {ls_time:.2f}s\n")

    print(f"SciPy Method:")
    print(f"  SNR: {scipy_snr:.2f} dB")
    print(f"  MSE: {scipy_mse:.2e}")
    print(f"  Max Difference: {scipy_diff_max:.2e}")
    print(f"  Mean Difference: {scipy_diff_mean:.2e}")
    print(f"  Time: {scipy_time:.2f}s\n")

    # Plot the original, resized, and difference images for both methods
    fig, ax = plt.subplots(2, 3, figsize=(18, 12))

    ax[0, 0].imshow(input_image_display, cmap="gray")
    ax[0, 0].set_title(f"Original Image {input_image_display.shape[0]} x {input_image_display.shape[1]}")
    ax[0, 0].axis("off")

    interp_order = {'Linear': 1, 'Quadratic': 2, 'Cubic': 3}[interpolation_type]

    ax[0, 1].imshow(ls_output_image_bg, cmap="gray")
    ax[0, 1].set_title(f"{method} Resized Image (Zoom: {zoom_factor}x, Interpolation: {interp_order}, Time: {ls_time:.2f}s)")
    ax[0, 1].axis("off")

    ax[0, 2].imshow(ls_difference_image_enhanced, cmap="gray", vmin=0, vmax=1)
    ax[0, 2].set_title(f"Difference Image (SNR: {ls_snr:.2f} dB, MSE: {ls_mse:.2e})")
    ax[0, 2].axis("off")

    ax[1, 0].imshow(input_image_display, cmap="gray")
    ax[1, 0].set_title(f"Original Image {input_image_display.shape[0]} x {input_image_display.shape[1]}")
    ax[1, 0].axis("off")

    ax[1, 1].imshow(scipy_output_image_bg, cmap="gray")
    ax[1, 1].set_title(f"SciPy Resized Image (Zoom: {zoom_factor}x, Interpolation: {interp_order}, Time: {scipy_time:.2f}s)")
    ax[1, 1].axis("off")

    ax[1, 2].imshow(scipy_difference_image_enhanced, cmap="gray", vmin=0, vmax=1)
    ax[1, 2].set_title(f"Difference Image (SNR: {scipy_snr:.2f} dB, MSE: {scipy_mse:.2e})")
    ax[1, 2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
