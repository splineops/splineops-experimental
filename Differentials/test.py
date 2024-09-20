import numpy as np
import matplotlib.pyplot as plt
from scipy.datasets import ascent
from Differentials import Differentials

# Load the "ascent" image from scipy datasets
image = ascent()

# Instantiate the Differentials class with the image
differential = Differentials(image)

# Choose an operation to run, e.g., Hessian Orientation
operation = Differentials.GRADIENT_MAGNITUDE

# Run the selected operation
differential.run(operation)

# The result is stored in differential.image
result = differential.image

# Define a mapping of operations to their names for titles
operation_titles = {
    Differentials.GRADIENT_MAGNITUDE: "Gradient Magnitude",
    Differentials.GRADIENT_DIRECTION: "Gradient Direction",
    Differentials.LAPLACIAN: "Laplacian",
    Differentials.LARGEST_HESSIAN: "Largest Hessian",
    Differentials.SMALLEST_HESSIAN: "Smallest Hessian",
    Differentials.HESSIAN_ORIENTATION: "Hessian Orientation"
}

# Display the original and the processed image
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title(operation_titles[operation])
plt.imshow(result, cmap='gray')

plt.show()
