import numpy as np
import matplotlib.pyplot as plt
from resize import resize_image  # Assuming resize.py contains the Resize class and resize_image function

def main():
    # Define original image dimensions
    input_height, input_width = 1000, 1000

    # Create a sample input image (e.g., a gradient for visualization purposes)
    input_image = np.linspace(0, 1, input_width).reshape(1, -1).repeat(input_height, axis=0)

    # Define desired output dimensions
    output_height, output_width = 500, 750  # Target output size

    # Resize the image to the specified output dimensions
    resized_image = resize_image(
        input_img_normalized=input_image,
        output_size=(output_height, output_width),
        interpolation="Cubic",
        inversable=False
    )

    # Plot original and resized images
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot original image
    axes[0].imshow(input_image, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f"Original Image ({input_width}x{input_height})")
    axes[0].axis('off')

    # Plot resized image
    axes[1].imshow(resized_image, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f"Resized Image ({output_width}x{output_height})")
    axes[1].axis('off')

    plt.show()

if __name__ == "__main__":
    main()
