# test_2d.py

import numpy as np
import matplotlib.pyplot as plt
from pyramidtools import get_pyramid_filter, reduce_2D, expand_2D

# Constants
NX = 4  # Size of the image (X Axis), should be even
NY = 4  # Size of the image (Y Axis), should be even

def main():
    # Get the filter coefficients for the Spline (order = 3) filter
    filter_name = "Spline"
    order = 3
    print(f"Filter: {filter_name}")
    g, h, is_centered = get_pyramid_filter(filter_name, order)
    print(f"Size of the reduce filter: {len(g)}")
    print(f"Size of the expand filter: {len(h)}")

    # Creation of a 2D input signal
    input_signal = np.array([
        [0.0, 1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 5.0],
        [3.0, 4.0, 3.0, 2.0]
    ])
    print("Input :")
    print(input_signal)

    # Reducing
    reduced_signal = reduce_2D(input_signal, g, is_centered)
    print("Reduced:")
    print(reduced_signal)

    # Expanding
    expanded_signal = expand_2D(reduced_signal, h, is_centered)
    print("Expanded :")
    print(expanded_signal)

    # Computing the error
    error_signal = expanded_signal[:NY, :NX] - input_signal
    print("Error :")
    print(error_signal)

    # Plotting the images
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(input_signal, cmap='gray', interpolation='nearest')
    plt.title('Original Image')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(reduced_signal, cmap='gray', interpolation='nearest')
    plt.title('Reduced Image')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(expanded_signal[:NY, :NX], cmap='gray', interpolation='nearest')
    plt.title('Expanded Image')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
