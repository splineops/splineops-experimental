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

# Resize and Compare Function
def test_resize_pattern(pattern, title, zoom_factors=(0.5, 0.5), interpolation='Cubic'):
    """Resizes the given pattern, displays it, and computes metrics between original and resized versions."""
    # Resize pattern
    resized_image = resize_image(pattern, zoom_factors=zoom_factors, interpolation=interpolation)
    
    # Metrics for comparison (since resized is a different size, metrics may vary)
    mse = np.mean((pattern[::2, ::2] - resized_image) ** 2)
    psnr = 10 * np.log10(1 / mse) if mse != 0 else float('inf')
    
    print(f"{title}")
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
    test_resize_pattern(gradient_image, "Gradient Image")

    # Test Sinusoidal Pattern
    sinusoidal_image = generate_sinusoidal_image(width, height, frequency=10)
    test_resize_pattern(sinusoidal_image, "Sinusoidal Image")

    # Test Checkerboard Pattern
    checkerboard_image = generate_checkerboard_image(width, height, square_size=100)
    test_resize_pattern(checkerboard_image, "Checkerboard Image")

if __name__ == "__main__":
    main()
