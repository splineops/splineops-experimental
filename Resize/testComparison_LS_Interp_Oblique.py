import os
from scipy.datasets import ascent
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import imageio.v2 as imageio
from Resize import Resize  # Importing the Resize class from Resize.py
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
    # Set degrees based on interpolation method
    if interpolation == "Linear":
        interp_degree = 1
        synthe_degree = 1
        analy_degree = 1
    elif interpolation == "Quadratic":
        interp_degree = 2
        synthe_degree = 2
        analy_degree = 2
    else:  # Cubic
        interp_degree = 3
        synthe_degree = 3
        analy_degree = 3

    # Interpolation method must fulfill requirement: analy_degree = -1
    # Least-Squares method must fulfill requirement: analy_degree = interp_degree
    # Oblique projection method must fulfill requirement: -1 < analy_degree < interp_degree

    if method == "Interpolation":
        analy_degree = -1
    elif method == "Oblique projection":
        if interpolation == "Linear":
            analy_degree = 3
        elif interpolation == "Quadratic":
            analy_degree = 1
        else:  # Cubic
            analy_degree = 1

    # Define the output image size
    output_height = int(np.round(input_image_normalized.shape[0] * zoom_factor))
    output_width = int(np.round(input_image_normalized.shape[1] * zoom_factor))
    output_image = np.zeros((output_height, output_width))

    # Create instance of Resize class
    resizer = Resize()

    # Perform resizing with a copy of the input image
    input_image_copy = input_image_normalized.copy()
    resizer.compute_zoom(
        input_image_copy,
        output_image,
        analy_degree=analy_degree,
        synthe_degree=synthe_degree,
        interp_degree=interp_degree,
        zoom_y=zoom_factor,
        zoom_x=zoom_factor,
        shift_y=0,
        shift_x=0,
        inversable=False,
    )

    # Define the reverse zoom output image size
    reverse_zoom_factor = 1.0 / zoom_factor
    reverse_output_height = int(np.round(output_image.shape[0] * reverse_zoom_factor))
    reverse_output_width = int(np.round(output_image.shape[1] * reverse_zoom_factor))
    reverse_output_image = np.zeros((reverse_output_height, reverse_output_width))

    # Perform reverse resizing with a copy of the output image
    output_image_copy = output_image.copy()
    resizer.compute_zoom(
        output_image_copy,
        reverse_output_image,
        analy_degree=analy_degree,
        synthe_degree=synthe_degree,
        interp_degree=interp_degree,
        zoom_y=reverse_zoom_factor,
        zoom_x=reverse_zoom_factor,
        shift_y=0,
        shift_x=0,
        inversable=False,
    )

    # Resize the reverse output image to match the original dimensions
    zoom_factors = (input_image_normalized.shape[0] / reverse_output_image.shape[0],
                    input_image_normalized.shape[1] / reverse_output_image.shape[1])
    resized_reverse_output_image = zoom(reverse_output_image, zoom_factors, order=interp_degree)

    # Compute SNR and MSE
    snr = compute_snr(input_image_normalized, resized_reverse_output_image)
    mse = compute_mse(input_image_normalized, resized_reverse_output_image)

    # Convert images to range [0, 255] for display
    output_image_display = np.clip(output_image * 255.0, 0, 255)
    resized_reverse_output_image_display = np.clip(resized_reverse_output_image * 255.0, 0, 255)

    return output_image_display, resized_reverse_output_image_display, snr, mse

def create_black_background(image, original_shape):
    # Create a black background with the original image dimensions
    black_background = np.zeros(original_shape)
    # Place the image in the top-left corner
    black_background[:image.shape[0], :image.shape[1]] = image
    return black_background

def main():
    # Load the ascent image
    #input_image = load_ascent_image()  # Use this to load the ascent image

    # Load chirp.tif image
    #input_image = load_image('collagen-mip.tif')

    # Load collagen-mip.tif image
    # input_image = load_image('chirp.tif')

    input_image = load_image('headMRI.tif')

    # Normalize the input image to [0, 1]
    input_image_normalized = input_image / 255.0

    # Define the zoom factor
    zoom_factor = 1.5

    # Define interpolation type
    interpolation_type = "Cubic"

    # Define the interpolation order
    interp_order = {'Linear': 1, 'Quadratic': 2, 'Cubic': 3}[interpolation_type]

    # Measure time and process images with least squares method
    methods = ["Least-Squares", "Interpolation", "Oblique projection"]
    results = []

    for method in methods:
        start_time = time.time()
        output_image, resized_reverse_output_image, snr, mse = resize_and_compute_snr(
            input_image_normalized, method, interpolation_type, zoom_factor
        )
        elapsed_time = time.time() - start_time
        results.append((method, output_image, resized_reverse_output_image, snr, mse, elapsed_time))

    # Convert the original image for display
    input_image_display = np.clip(input_image_normalized * 255.0, 0, 255)

    # Create black backgrounds if zoom factor is less than 1.0
    for i, (method, output_image, resized_reverse_output_image, snr, mse, elapsed_time) in enumerate(results):
        if zoom_factor < 1.0:
            output_image_bg = create_black_background(output_image, input_image_display.shape)
            results[i] = (method, output_image_bg, resized_reverse_output_image, snr, mse, elapsed_time)

    # Compute the difference image for the reversed images
    enhancement_factor = 5
    difference_images = []
    for method, output_image, resized_reverse_output_image, snr, mse, elapsed_time in results:
        difference_image = np.abs(input_image_normalized - resized_reverse_output_image / 255.0) * enhancement_factor
        diff_max = np.max(difference_image)
        diff_mean = np.mean(difference_image)
        difference_images.append((method, output_image, difference_image, snr, mse, elapsed_time, diff_max, diff_mean))

    # Print statistics to the terminal
    for method, output_image, difference_image, snr, mse, elapsed_time, diff_max, diff_mean in difference_images:
        print(f"{method} Method:")
        print(f"  SNR: {snr:.2f} dB")
        print(f"  MSE: {mse:.2e}")
        print(f"  Max Difference: {diff_max:.2e}")
        print(f"  Mean Difference: {diff_mean:.2e}")
        print(f"  Time: {elapsed_time:.2f}s\n")

    # Plot the original, resized, and difference images for each method
    fig, ax = plt.subplots(3, 3, figsize=(18, 18))

    for i, (method, output_image, difference_image, snr, mse, elapsed_time, _, _) in enumerate(difference_images):
        ax[i, 0].imshow(input_image_display, cmap="gray")
        ax[i, 0].set_title(f"Original Image {input_image_display.shape[0]} x {input_image_display.shape[1]}")
        ax[i, 0].axis("off")

        ax[i, 1].imshow(output_image, cmap="gray")
        ax[i, 1].set_title(f"{method} Resized Image (Zoom: {zoom_factor}x, Interpolation: {interp_order}, Time: {elapsed_time:.2f}s)")
        ax[i, 1].axis("off")

        ax[i, 2].imshow(difference_image, cmap="gray", vmin=0, vmax=1)
        ax[i, 2].set_title(f"Difference Image (SNR: {snr:.2f} dB, MSE: {mse:.2e})")
        ax[i, 2].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
