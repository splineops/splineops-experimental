import numpy as np
import matplotlib.pyplot as plt
from Resize_No_Vectorization import Resize

# Create a simple gradient image for boundary analysis
def create_gradient_image(size):
    """Create a grayscale gradient image to test boundary effects."""
    return np.tile(np.linspace(0, 255, size[1]), (size[0], 1))

def resize_with_diagnostic(input_image, degrees, zoom_factors=(0.3, 0.3)):
    resizer = Resize()
    output_size = (
        int(input_image.shape[0] * zoom_factors[0]),
        int(input_image.shape[1] * zoom_factors[1])
    )
    output_image = np.zeros(output_size)
    
    # Normalize input image for processing
    normalized_input = input_image / 255.0

    print(f"\n--- Diagnostic for Degrees: {degrees} ---")

    # Run compute_zoom and capture output
    resizer.compute_zoom(
        input_img=normalized_input, 
        output_img=output_image, 
        analy_degree=degrees[0], 
        synthe_degree=degrees[1], 
        interp_degree=degrees[2], 
        zoom_y=zoom_factors[0], 
        zoom_x=zoom_factors[1], 
        shift_y=0, 
        shift_x=0, 
        inversable=False
    )
    
    # Output resized image
    return output_image

def plot_results(original_img, output_images, titles):
    """Plot original and resized images with different spline degrees."""
    plt.figure(figsize=(15, 8))
    plt.subplot(2, len(output_images), 1)
    plt.imshow(original_img, cmap='gray')
    plt.title("Original Gradient")
    
    for i, (output_img, title) in enumerate(zip(output_images, titles)):
        plt.subplot(2, len(output_images), i + 2)
        plt.imshow(output_img, cmap='gray', vmin=0, vmax=1)
        plt.title(title)
    
    plt.tight_layout()
    plt.show()

def experiment_with_border_adjustments(input_image, degrees, zoom_factors=(0.3, 0.3)):
    """Test different border adjustments for quadratic interpolation (degree 2)."""
    output_images = []
    titles = []
    
    for extra_border in [0, 2, 4, 6, 8]:
        resizer = Resize()
        output_size = (
            int(input_image.shape[0] * zoom_factors[0]),
            int(input_image.shape[1] * zoom_factors[1])
        )
        output_image = np.zeros(output_size)
        normalized_input = input_image / 255.0
        
        print(f"\n--- Experimenting with Extra Border: {extra_border} ---")
        
        # Temporarily override border calculation within this instance
        resizer.border = lambda size, degree: Resize.border(size, degree) + extra_border
        
        # Run compute_zoom with modified border
        resizer.compute_zoom(
            input_img=normalized_input, 
            output_img=output_image, 
            analy_degree=degrees[0], 
            synthe_degree=degrees[1], 
            interp_degree=degrees[2], 
            zoom_y=zoom_factors[0], 
            zoom_x=zoom_factors[1], 
            shift_y=0, 
            shift_x=0, 
            inversable=False
        )
        
        output_images.append(output_image)
        titles.append(f"Extra Border: {extra_border}")
    
    plot_results(input_image, output_images, titles)

# Main execution
if __name__ == "__main__":
    # Generate a gradient image to simplify boundary inspection
    gradient_img = create_gradient_image((100, 100))
    
    # Degrees to test: Linear, Quadratic, and Cubic
    degrees_list = [(1, 1, 1), (2, 2, 2), (3, 3, 3)]
    #degrees_list = [(2, 2, 2)]
    zoom_factors = (0.3, 0.3)

    # Step 1: Test resizing with different degrees
    output_images = []
    titles = []
    for degrees in degrees_list:
        resized_img = resize_with_diagnostic(gradient_img, degrees, zoom_factors)
        output_images.append(resized_img)
        titles.append(f"Degrees: {degrees}")

    # Plot results of different degrees
    plot_results(gradient_img, output_images, titles)

    # Step 2: Experiment with border adjustments for quadratic interpolation
    experiment_with_border_adjustments(gradient_img, degrees=(2, 2, 2), zoom_factors=zoom_factors)
