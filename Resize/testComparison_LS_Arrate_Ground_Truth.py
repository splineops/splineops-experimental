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

def load_ascent_image():
    img = ascent()  # Load the ascent image from scipy
    img_resized = zoom(img, (256 / img.shape[0], 256 / img.shape[1]), order=3)  # Resize to 256x256
    return img_resized  # Keep the image values in the range [0, 255]

def load_image(filename):
    img_path = os.path.join(os.path.dirname(__file__), 'Samples', filename)
    img = imageio.imread(img_path)
    if len(img.shape) == 3 and img.shape[2] == 4:  # If the image has 4 channels (RGBA)
        img = np.mean(img[:, :, :3], axis=2)  # Ignore the alpha channel and average RGB channels
    img_resized = zoom(img, (256 / img.shape[0], 256 / img.shape[1]), order=3)  # Resize to 256x256

    if img_resized.dtype != np.uint8:
        img_resized = np.clip(img_resized, 0, 255).astype(np.uint8)
    
    output_filename = os.path.splitext(filename)[0] + '_256x256' + os.path.splitext(filename)[1]
    output_path = os.path.join(os.path.dirname(__file__), 'Samples', output_filename)
    imageio.imwrite(output_path, img_resized)

    return img_resized  # Keep the image values in the range [0, 255]

def load_ground_truth(filename):
    img_path = os.path.join(os.path.dirname(__file__), 'Samples', filename)
    img = imageio.imread(img_path)
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = np.mean(img[:, :, :3], axis=2)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def compute_snr(original, processed):
    signal_power = 255.0 ** 2
    noise_power = np.mean((original - processed) ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def compute_mse(original, processed):
    mse = np.mean((original - processed) ** 2)
    return mse

def resize_and_compute_snr(input_image_normalized, method, interpolation, zoom_factor):
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

    if method == "Interpolation":
        analy_degree = -1
    elif method == "Oblique projection":
        if interpolation == "Linear":
            analy_degree = 0
        elif interpolation == "Quadratic":
            analy_degree = 1
        else:  # Cubic
            analy_degree = 2

    output_height = int(np.round(input_image_normalized.shape[0] * zoom_factor))
    output_width = int(np.round(input_image_normalized.shape[1] * zoom_factor))
    output_image = np.zeros((output_height, output_width))

    resizer = Resize()

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

    reverse_zoom_factor = 1.0 / zoom_factor
    reverse_output_height = int(np.round(output_image.shape[0] * reverse_zoom_factor))
    reverse_output_width = int(np.round(output_image.shape[1] * reverse_zoom_factor))
    reverse_output_image = np.zeros((reverse_output_height, reverse_output_width))

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

    zoom_factors = (input_image_normalized.shape[0] / reverse_output_image.shape[0],
                    input_image_normalized.shape[1] / reverse_output_image.shape[1])
    resized_reverse_output_image = zoom(reverse_output_image, zoom_factors, order=interp_degree)

    snr = compute_snr(input_image_normalized, resized_reverse_output_image)
    mse = compute_mse(input_image_normalized, resized_reverse_output_image)

    output_image_display = np.clip(output_image * 255.0, 0, 255)
    resized_reverse_output_image_display = np.clip(resized_reverse_output_image * 255.0, 0, 255)

    return output_image_display, resized_reverse_output_image_display, snr, mse

def create_black_background(image, original_shape):
    black_background = np.zeros(original_shape)
    black_background[:image.shape[0], :image.shape[1]] = image
    return black_background

def main():
    input_image = load_image('headMRI.tif')
    ground_truth_image_1500_normalized = load_ground_truth('headMRI_1500x1500_Ground_Truth.tif') / 255.0
    ground_truth_image_256_normalized = load_ground_truth('headMRI_256x256_Converted_Ground_Truth.tif') / 255.0

    input_image_normalized = input_image / 255.0

    zoom_factor = 1.5
    method = "Least-Squares"
    interpolation_type = "Cubic"

    start_time = time.time()
    ls_output_image, ls_resized_reverse_output_image, ls_snr, ls_mse = resize_and_compute_snr(
        input_image_normalized, method, interpolation_type, zoom_factor
    )
    ls_time = time.time() - start_time

    ground_truth_snr = compute_snr(input_image_normalized, ground_truth_image_256_normalized)
    ground_truth_mse = compute_mse(input_image_normalized, ground_truth_image_256_normalized)

    input_image_display = np.clip(input_image_normalized * 255.0, 0, 255)

    if zoom_factor < 1.0:
        ls_output_image_bg = create_black_background(ls_output_image, input_image_display.shape)
    else:
        ls_output_image_bg = ls_output_image

    enhancement_factor = 100
    ls_difference_image = np.abs(input_image_normalized - ls_resized_reverse_output_image / 255.0)
    ls_diff_max = np.max(ls_difference_image)
    ls_diff_mean = np.mean(ls_difference_image)

    ls_difference_image_enhanced = ls_difference_image * enhancement_factor
    ls_difference_image_enhanced = np.clip(ls_difference_image_enhanced, 0, 1)

    ground_truth_image_diff = np.abs(input_image_normalized - ground_truth_image_256_normalized)
    ground_truth_image_diff_max = np.max(ground_truth_image_diff)
    ground_truth_image_diff_mean = np.mean(ground_truth_image_diff)
    ground_truth_image_enhanced = np.clip(ground_truth_image_diff * enhancement_factor, 0, 1)


    print(f"Least-Squares Method:")
    print(f"  SNR: {ls_snr:.2f} dB")
    print(f"  MSE: {ls_mse:.2e}")
    print(f"  Max Difference: {ls_diff_max:.2e}")
    print(f"  Mean Difference: {ls_diff_mean:.2e}")
    print(f"  Time: {ls_time:.2f}s\n")

    print(f"Ground Truth Method:")
    print(f"  SNR: {ground_truth_snr:.2f} dB")
    print(f"  MSE: {ground_truth_mse:.2e}")
    print(f"  Max Difference: {ground_truth_image_diff_max:.2e}")
    print(f"  Mean Difference: {ground_truth_image_diff_mean:.2e}")

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

    ax[1, 1].imshow(ground_truth_image_1500_normalized, cmap="gray")
    ax[1, 1].set_title(f"Ground Truth Image (Resized to 256x256)")
    ax[1, 1].axis("off")

    ax[1, 2].imshow(ground_truth_image_enhanced, cmap="gray", vmin=0, vmax=1)
    ax[1, 2].set_title(f"Difference Image (Enhanced)")
    ax[1, 2].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
