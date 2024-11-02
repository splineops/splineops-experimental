import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom as scipy_zoom
from Resize_No_Vectorization import Resize  # Assuming Resize class from external module

def resize_image_with_least_squares(scale_factor):
    """
    Resizes a square image by scaling up using the custom Resize class and then
    scales it back down to the original size.

    Parameters
    ----------
    scale_factor : int
        Integer scaling factor for both vertical and horizontal upscaling.
    """
    # Create a 10x10 square image
    input_img = np.zeros((10, 10))
    input_img[3:7, 3:7] = 255.0  # White square in the center

    # Normalize the input image to the range [0, 1]
    input_img_normalized = input_img / 255.0
    input_img_normalized_copy = input_img_normalized.copy()

    # Define output shape for upscaled image based on scaling factor
    output_shape = (input_img.shape[0] * scale_factor, input_img.shape[1] * scale_factor)
    upscaled_img = np.zeros(output_shape, dtype=np.float64)

    # Initialize Resize instance
    resizer = Resize()
    
    # Set parameters for least-squares cubic interpolation
    interp_degree = 3  # Cubic interpolation
    synthe_degree = 3
    analy_degree = 3
    shift_y = 0.0  # No shift
    shift_x = 0.0
    inversable = False  # Non-inversible for this example

    # Step 1: Upscale to the larger size using the custom Resize class
    resizer.compute_zoom(
        input_img=input_img_normalized, 
        output_img=upscaled_img, 
        analy_degree=analy_degree, 
        synthe_degree=synthe_degree, 
        interp_degree=interp_degree, 
        zoom_y=scale_factor, 
        zoom_x=scale_factor, 
        shift_y=shift_y, 
        shift_x=shift_x, 
        inversable=inversable
    )

    upscaled_img_copy = upscaled_img.copy()

    # Upscale with scipy.ndimage.zoom for comparison
    upscaled_scipy = scipy_zoom(input_img_normalized, (scale_factor, scale_factor), order=3)

    # Step 2: Resize back to 10x10 using custom Resize class
    reverted_shape = (10, 10)
    reverted_img = np.zeros(reverted_shape, dtype=np.float64)
    reverse_zoom_y = 1.0 / scale_factor
    reverse_zoom_x = 1.0 / scale_factor

    resizer.compute_zoom(
        input_img=upscaled_img, 
        output_img=reverted_img, 
        analy_degree=analy_degree, 
        synthe_degree=synthe_degree, 
        interp_degree=interp_degree, 
        zoom_y=reverse_zoom_y, 
        zoom_x=reverse_zoom_x, 
        shift_y=shift_y, 
        shift_x=shift_x, 
        inversable=inversable
    )

    # Revert using scipy.ndimage.zoom for comparison
    reverted_scipy = scipy_zoom(upscaled_scipy, (reverse_zoom_y, reverse_zoom_x), order=3)

    # Plotting
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    
    # Original input
    axs[0, 0].imshow(input_img_normalized_copy, cmap='gray')
    axs[0, 0].set_title("Original Image")

    # Upscaled images
    axs[1, 0].imshow(upscaled_img_copy, cmap='gray')
    axs[1, 0].set_title(f"Upscaled Image (Custom, {output_shape[0]}x{output_shape[1]})")

    axs[1, 1].imshow(upscaled_scipy, cmap='gray')
    axs[1, 1].set_title("Upscaled Image (SciPy)")

    # Reverted images
    axs[2, 0].imshow(reverted_img, cmap='gray')
    axs[2, 0].set_title("Reverted Image (Custom)")

    axs[2, 1].imshow(reverted_scipy, cmap='gray')
    axs[2, 1].set_title("Reverted Image (SciPy)")

    for ax in axs.flat:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage with 10x10 square image:
resize_image_with_least_squares(scale_factor=2)
