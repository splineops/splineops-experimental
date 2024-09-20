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
    #input_image_filename = 'collagen-mip.tif'
    #input_image = load_image(input_image_filename)

    # Load collagen-mip.tif image
    # input_image_filename = 'chirp.tif'
    # input_image = load_image(input_image_filename)

    input_image_filename = 'headMRI.tif'

    #input_image_filename = 'chirp.tif'

    input_image = load_image(input_image_filename)

    # Normalize the input image to [0, 1]
    input_image_normalized = input_image / 255.0

    # Define interpolation type
    interpolation_type = "Linear"

    # Define the interpolation order
    interp_order = {'Linear': 1, 'Quadratic': 2, 'Cubic': 3}[interpolation_type]

    # Iterate over several zoom factors, excluding 1.0
    zoom_factors = np.linspace(0.1, 2.0, 30)
    zoom_factors = [zf for zf in zoom_factors if not np.isclose(zf, 1.0)]

    methods = ["Least-Squares", "Interpolation", "Oblique projection"]
    results = {method: {'snrs': [], 'mses': []} for method in methods}

    for zoom_factor in zoom_factors:
        for method in methods:
            _, _, snr, mse = resize_and_compute_snr(
                input_image_normalized, method, interpolation_type, zoom_factor
            )
            results[method]['snrs'].append(snr)
            results[method]['mses'].append(mse)

    # Plot SNR for each method
    plt.figure(figsize=(12, 6))
    for method in methods:
        plt.plot(zoom_factors, results[method]['snrs'], label=f'{method} Method SNR', marker='o')
    plt.xlabel('Zoom Factor')
    plt.ylabel('SNR (dB)')
    plt.title(f'SNR Comparison for {input_image_filename} with {interpolation_type} Interpolation')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot logarithmic MSE for each method
    plt.figure(figsize=(12, 6))
    for method in methods:
        plt.plot(zoom_factors, np.log10(results[method]['mses']), label=f'{method} Method log(MSE)', marker='o')
    plt.xlabel('Zoom Factor')
    plt.ylabel('log(MSE)')
    plt.title(f'Logarithmic MSE Comparison for {input_image_filename} with {interpolation_type} Interpolation')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
