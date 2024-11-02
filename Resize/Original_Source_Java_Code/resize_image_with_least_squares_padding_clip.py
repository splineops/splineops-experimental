import numpy as np
import matplotlib.pyplot as plt
from Resize_No_Vectorization import Resize  # Assuming Resize class from external module

def resize_image_with_least_squares(zoom_y, zoom_x):
    # Step 1: Create the initial 10x10 square image
    input_img = np.zeros((10, 10))
    input_img[3:7, 3:7] = 255.0  # White square in the center
    
    # Normalize the input image to the range [0, 1]
    input_img_normalized = input_img / 255.0
    input_img_normalized_copy = input_img_normalized.copy()

    # Step 2: Downscale the image to 5x5
    output_shape = (int(round(input_img.shape[0] * zoom_y)), int(round(input_img.shape[1] * zoom_x)))
    downscaled_img = np.zeros(output_shape, dtype=np.float64)
    resizer = Resize()
    
    # Set parameters for least-squares cubic interpolation
    interp_degree = 3  # Cubic interpolation
    synthe_degree = 3
    analy_degree = 3
    shift_y = 0.0  # No shift
    shift_x = 0.0
    inversable = False  # Non-inversible for this example

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
    downscaled_img = np.clip(downscaled_img, 0, 1)  # Clip values to [0, 1]

    # Step 3: Pad the 5x5 downscaled image to make it 10x10
    padded_img = np.zeros((10, 10), dtype=np.float64)
    start_y = (10 - downscaled_img.shape[0]) // 2
    start_x = (10 - downscaled_img.shape[1]) // 2
    padded_img[start_y:start_y+downscaled_img.shape[0], start_x:start_x+downscaled_img.shape[1]] = downscaled_img
    padded_img_copy = padded_img.copy()  # Save a copy before upsampling

    # Step 4: Upscale the padded 10x10 image by a factor of 2 to get a 20x20 image
    upscaled_img = np.zeros((20, 20), dtype=np.float64)
    resizer.compute_zoom(
        input_img=padded_img, 
        output_img=upscaled_img, 
        analy_degree=analy_degree, 
        synthe_degree=synthe_degree, 
        interp_degree=interp_degree, 
        zoom_y=2.0, 
        zoom_x=2.0, 
        shift_y=shift_y, 
        shift_x=shift_x, 
        inversable=inversable
    )
    upscaled_img = np.clip(upscaled_img, 0, 1)  # Clip values to [0, 1]

    # Step 5: Extract the central 10x10 region from the 20x20 upscaled image
    final_img = upscaled_img[5:15, 5:15]
    final_img = np.clip(final_img, 0, 1)  # Clip values to [0, 1]

    # Plotting the results
    fig, axs = plt.subplots(4, 1, figsize=(6, 16))
    
    axs[0].imshow(input_img_normalized_copy, cmap='gray')
    axs[0].set_title("Initial Square Image (Python)")
    
    axs[1].imshow(downscaled_img, cmap='gray')
    axs[1].set_title("Downscaled Image (5x5, Python)")
    
    axs[2].imshow(padded_img_copy, cmap='gray')
    axs[2].set_title("Padded Image (10x10 with 5x5 centered, Python)")
    
    axs[3].imshow(final_img, cmap='gray')
    axs[3].set_title("Final Reverted Image (10x10 extracted from 20x20, Python)")
    
    for ax in axs.flat:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage with 10x10 square image:
resize_image_with_least_squares(
    zoom_y=0.5,  # Vertical zoom factor
    zoom_x=0.5   # Horizontal zoom factor
)
