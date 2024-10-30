import numpy as np
from scipy.datasets import ascent
import matplotlib.pyplot as plt
from PIL import Image
from Resize_No_Vectorization import Resize

def resize_image_with_quadratic(zoom_y, zoom_x):
    # Load the ascent image from scipy
    original_img = ascent()
    input_img = original_img.copy()  # Make a copy to use as input for resizing
    
    # Normalize the input image to the range [0, 1]
    input_img_normalized = input_img / 255.0

    # Define output shape based on zoom factors
    output_shape = (int(round(input_img.shape[0] * zoom_y)), int(round(input_img.shape[1] * zoom_x)))
    output_img = np.zeros(output_shape, dtype=np.float64)

    # Initialize Resize instance
    resizer = Resize()
    
    # Set parameters for quadratic interpolation
    interp_degree = 1  # Quadratic
    synthe_degree = 1
    analy_degree = 1
    shift_y = 0.0  # No shift
    shift_x = 0.0
    inversable = False  # Set according to preference

    # Perform the resize operation
    resizer.compute_zoom(
        input_img=input_img_normalized, 
        output_img=output_img, 
        analy_degree=analy_degree, 
        synthe_degree=synthe_degree, 
        interp_degree=interp_degree, 
        zoom_y=zoom_y, 
        zoom_x=zoom_x, 
        shift_y=shift_y, 
        shift_x=shift_x, 
        inversable=inversable
    )

    # Convert the output to an image format
    output_img = np.clip(output_img * 255, 0, 255).astype(np.uint8)  # Denormalize and convert to uint8
    output_image = Image.fromarray(output_img)

    # Display the images side-by-side for comparison
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Ascent Image")
    plt.imshow(original_img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Resized Image (Quadratic, Zoom 0.3x0.3)")
    plt.imshow(output_img, cmap='gray')
    plt.axis('off')
    plt.show()

# Example usage with ascent image:
resize_image_with_quadratic(
    zoom_y=0.3,  # Vertical zoom factor
    zoom_x=0.3   # Horizontal zoom factor
)
