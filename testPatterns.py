import numpy as np
import matplotlib.pyplot as plt
from resize import resize_image

# Pattern Generators
def generate_gradient_image(width, height):
    """Generates a gradient image with pixel values increasing linearly from left to right."""
    return np.linspace(0, 1, width).reshape(1, -1).repeat(height, axis=0)

def generate_sinusoidal_image(width, height, frequency=5):
    """Generates a sinusoidal pattern along the horizontal axis."""
    x = np.linspace(0, 2 * np.pi * frequency, width)
    y = np.sin(x) * 0.5 + 0.5  # Normalize to [0, 1]
    return np.tile(y, (height, 1))

def generate_checkerboard_image(width, height, square_size=20):
    """Generates a checkerboard pattern with alternating blocks."""
    rows = (np.arange(height) // square_size) % 2
    cols = (np.arange(width) // square_size) % 2
    return np.bitwise_xor.outer(rows, cols).astype(float)

def calculate_mse(original, resized, zoom_factors):
    # Calculate the target size based on zoom factors
    target_height = int(original.shape[0] * zoom_factors[0])
    target_width = int(original.shape[1] * zoom_factors[1])
    
    # Downsample the original to match the target size
    downsampled_original = original[:target_height * int(1 / zoom_factors[0]):int(1 / zoom_factors[0]), 
                                    :target_width * int(1 / zoom_factors[1]):int(1 / zoom_factors[1])]
    
    # Calculate MSE between the downsampled original and resized image
    mse = np.mean((downsampled_original - resized) ** 2)
    return mse


# Resize and Compare Function
def test_resize_pattern(pattern, title, zoom_factors=(0.5, 0.5), interpolation='Cubic'):
    """Resizes the given pattern, displays it, and computes metrics between original and resized versions."""
    # Resize pattern
    resized_image = resize_image(pattern, zoom_factors=zoom_factors, interpolation=interpolation)
    
    # Calculate MSE and PSNR based on arbitrary zoom factors
    mse = calculate_mse(pattern, resized_image, zoom_factors)
    psnr = 10 * np.log10(1 / mse) if mse != 0 else float('inf')
    
    print(f"{title} with zoom factors {zoom_factors}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Peak Signal-to-Noise Ratio (PSNR): {psnr} dB\n")
    
    # Plot original and resized images
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(pattern, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f"Original {title}")
    axes[0].axis('off')
    
    axes[1].imshow(resized_image, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f"Resized {title}")
    axes[1].axis('off')
    
    plt.show()

# Main test function
def main():
    width, height = 1000, 1000
    print("Testing on Gradient, Sinusoidal, and Checkerboard Patterns\n")

    # Test Gradient Pattern
    gradient_image = generate_gradient_image(width, height)
    test_resize_pattern(gradient_image, "Gradient Image", zoom_factors=(0.75, 0.75))

    # Test Sinusoidal Pattern
    sinusoidal_image = generate_sinusoidal_image(width, height, frequency=10)
    test_resize_pattern(sinusoidal_image, "Sinusoidal Image", zoom_factors=(0.5, 0.5))

    # Test Checkerboard Pattern
    checkerboard_image = generate_checkerboard_image(width, height, square_size=100)
    test_resize_pattern(checkerboard_image, "Checkerboard Image", zoom_factors=(0.3, 0.3))

if __name__ == "__main__":
    main()
