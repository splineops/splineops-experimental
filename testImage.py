import numpy as np
import matplotlib.pyplot as plt
from resize import resize_image

# Mathematical functions for expected values in each pattern

def expected_gradient_value(x, width):
    """Expected value at position x for a gradient pattern of given width."""
    return x / width

def expected_sinusoidal_value(x, width, frequency=5):
    """Expected value at position x for a sinusoidal pattern of given width."""
    normalized_x = x / width * 2 * np.pi * frequency
    return np.sin(normalized_x) * 0.5 + 0.5

def expected_checkerboard_value(x, y, square_size):
    """Expected value at position (x, y) for a checkerboard pattern."""
    row, col = int(y // square_size), int(x // square_size)
    return (row + col) % 2

# Function to calculate the MSE using expected values
def calculate_mse_with_expected(pattern_name, width, height, zoom_factors, resized_image):
    """Calculates MSE between the resized image and mathematically expected values."""
    target_height, target_width = resized_image.shape
    
    # Calculate expected values based on pattern type
    if pattern_name == "Gradient":
        expected_values = np.array([[expected_gradient_value(x / zoom_factors[1], width)
                                     for x in range(target_width)]
                                     for y in range(target_height)])
    elif pattern_name == "Sinusoidal":
        expected_values = np.array([[expected_sinusoidal_value(x / zoom_factors[1], width, frequency=10)
                                     for x in range(target_width)]
                                     for y in range(target_height)])
    elif pattern_name == "Checkerboard":
        expected_values = np.array([[expected_checkerboard_value(x / zoom_factors[1], y / zoom_factors[0], 100)
                                     for x in range(target_width)]
                                     for y in range(target_height)])
    else:
        raise ValueError("Unknown pattern name")
    
    # Calculate MSE
    mse = np.mean((expected_values - resized_image) ** 2)
    return mse

# Resize and Compare Function
def test_resize_pattern(pattern_name, width, height, zoom_factors=(0.5, 0.5), interpolation='Cubic'):
    """Generates a pattern, resizes it, and compares with mathematically expected values."""
    # Generate pattern and resize
    if pattern_name == "Gradient":
        pattern = np.linspace(0, 1, width).reshape(1, -1).repeat(height, axis=0)
    elif pattern_name == "Sinusoidal":
        x = np.linspace(0, 2 * np.pi * 10, width)
        y = np.sin(x) * 0.5 + 0.5
        pattern = np.tile(y, (height, 1))
    elif pattern_name == "Checkerboard":
        rows = (np.arange(height) // 100) % 2
        cols = (np.arange(width) // 100) % 2
        pattern = np.bitwise_xor.outer(rows, cols).astype(float)
    else:
        raise ValueError("Unknown pattern name")
    
    # Resize pattern
    resized_image = resize_image(pattern, zoom_factors=zoom_factors, interpolation=interpolation)
    
    # Calculate MSE and PSNR with expected values
    mse = calculate_mse_with_expected(pattern_name, width, height, zoom_factors, resized_image)
    psnr = 10 * np.log10(1 / mse) if mse != 0 else float('inf')
    
    print(f"{pattern_name} with zoom factors {zoom_factors}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Peak Signal-to-Noise Ratio (PSNR): {psnr} dB\n")
    
    # Plot original and resized images
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(pattern, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f"Original {pattern_name}")
    axes[0].axis('off')
    
    axes[1].imshow(resized_image, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f"Resized {pattern_name}")
    axes[1].axis('off')
    
    plt.show()

# Main test function
def main():
    width, height = 1000, 1000
    print("Testing on Gradient, Sinusoidal, and Checkerboard Patterns\n")

    # Test Gradient Pattern
    test_resize_pattern("Gradient", width, height, zoom_factors=(0.75, 0.5))

    # Test Sinusoidal Pattern
    test_resize_pattern("Sinusoidal", width, height, zoom_factors=(0.5, 0.5))

    # Test Checkerboard Pattern
    test_resize_pattern("Checkerboard", width, height, zoom_factors=(0.3, 0.6))

if __name__ == "__main__":
    main()
