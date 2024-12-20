import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom as scipy_zoom
from Resize_No_Vectorization import Resize  # Assuming Resize class from external module

def resize_image_with_least_squares(zoom_y, zoom_x):
    # Create a 10x10 square image
    input_img = np.zeros((10, 10))
    input_img[3:7, 3:7] = 255.0  # White square in the center
    
    # Normalize the input image to the range [0, 1]
    input_img_normalized = input_img / 255.0

    input_img_normalized_copy = input_img_normalized.copy()

    # Define output shape for downscaled image based on zoom factors
    output_shape = (int(round(input_img.shape[0] * zoom_y)), int(round(input_img.shape[1] * zoom_x)))
    downscaled_img = np.zeros(output_shape, dtype=np.float64)

    # Initialize Resize instance
    resizer = Resize()
    
    # Set parameters for least-squares cubic interpolation
    interp_degree = 3  # Cubic interpolation
    synthe_degree = 3
    analy_degree = 3
    shift_y = 0.0  # No shift
    shift_x = 0.0
    inversable = False  # Non-inversible for this example

    # Step 1: Downscale to smaller size using custom Resize class
    resizer.compute_zoom(
        input_img=input_img_normalized, 
        output_img=downscaled_img, 
        analy_degree=analy_degree, 
        synthe_degree=synthe_degree, 
        interp_degree=interp_degree, 
        zoom_y=zoom_y, 
        zoom_x=zoom_x, 
        shift_y=shift_y, 
        shift_x=shift_x, 
        inversable=inversable
    )

    # Re-map the downscaled image to range [0, 1]
    downscaled_img = (downscaled_img - downscaled_img.min()) / (downscaled_img.max() - downscaled_img.min())
    downscaled_img_copy = downscaled_img.copy()

    # Step 2: Resize back to original size using custom Resize class
    reverted_shape = (10, 10)
    reverted_img = np.zeros(reverted_shape, dtype=np.float64)
    reverse_zoom_y = 1.0 / zoom_y
    reverse_zoom_x = 1.0 / zoom_x

    resizer.compute_zoom(
        input_img=downscaled_img, 
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

    # Re-map the reverted image to range [0, 1]
    reverted_img = (reverted_img - reverted_img.min()) / (reverted_img.max() - reverted_img.min())

    # Downscale and revert using scipy.ndimage.zoom for comparison
    downscaled_scipy = scipy_zoom(input_img_normalized, (zoom_y, zoom_x), order=3)
    reverted_scipy = scipy_zoom(downscaled_scipy, (reverse_zoom_y, reverse_zoom_x), order=3)

    # Plotting
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    
    # Original input
    axs[0, 0].imshow(input_img_normalized_copy, cmap='gray')
    axs[0, 0].set_title("Original Image")

    # Downscaled images
    axs[1, 0].imshow(downscaled_img_copy, cmap='gray')
    axs[1, 0].set_title(f"Downscaled Image (Custom, {output_shape[0]}x{output_shape[1]})")

    axs[1, 1].imshow(downscaled_scipy, cmap='gray')
    axs[1, 1].set_title("Downscaled Image (SciPy)")

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
resize_image_with_least_squares(
    zoom_y=0.5,  # Vertical zoom factor
    zoom_x=0.5   # Horizontal zoom factor
)
