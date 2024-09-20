# test_1d.py

import numpy as np
import matplotlib.pyplot as plt
from pyramidtools import get_pyramid_filter, reduce_1D, expand_1D

# Constants
LENGTH = 10  # Length of the input signal, should be even

# Main function
def main():
    # Get the filter coefficients for the Spline (order = 3) filter
    filter_name = "Centered Spline"
    order = 3
    print(f"Filter: {filter_name}")
    g, h, is_centered = get_pyramid_filter(filter_name, order)
    print(f"Size of the reduce filter: {len(g)}")
    print(f"Size of the expand filter: {len(h)}")

    # Creation of a 1D input signal
    input_signal = np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, -2.0, -4.0, -6.0])
    print("Input   :", " ".join(f"{v:2.3f}" for v in input_signal))

    # Reducing
    reduced_signal = reduce_1D(input_signal, g, is_centered)
    print("Reduced :", " ".join(f"{v:2.3f}" for v in reduced_signal))

    # Expanding
    expanded_signal = expand_1D(reduced_signal, h, is_centered)
    print("Expanded:", " ".join(f"{v:2.3f}" for v in expanded_signal))

    # Computing the error
    error_signal = expanded_signal[:LENGTH] - input_signal
    print("Error   :", " ".join(f"{v:2.3f}" for v in error_signal))

    # Plotting the signals
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.stem(range(len(input_signal)), input_signal, basefmt=" ")
    plt.title('Original Signal')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')

    plt.subplot(3, 1, 2)
    plt.stem(range(len(reduced_signal)), reduced_signal, basefmt=" ")
    plt.title('Reduced Signal')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')

    plt.subplot(3, 1, 3)
    plt.stem(range(len(expanded_signal[:LENGTH])), expanded_signal[:LENGTH], basefmt=" ")
    plt.title('Expanded Signal')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
