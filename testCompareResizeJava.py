import numpy as np
import csv
from resize import resize_image

# Load the Java-generated resized image data from CSV
def load_java_output(filepath):
    with open(filepath, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        java_resized = np.array([[float(pixel) for pixel in row if pixel] for row in reader])
    return java_resized

# Load the shared input image from CSV
def load_input_image(filepath):
    with open(filepath, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        input_image = np.array([[float(pixel) for pixel in row if pixel] for row in reader])
    return input_image

# Compare two images using Mean Squared Error (MSE) and Peak Signal-to-Noise Ratio (PSNR)
def calculate_metrics(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    psnr = 10 * np.log10(1 / mse) if mse != 0 else float('inf')
    return mse, psnr

# Main function
def main():
    # Filepaths
    java_output_filepath = 'java_output.csv'
    input_image_filepath = 'input_image.csv'
    
    # Load Java resized image and shared input image
    java_resized_image = load_java_output(java_output_filepath)
    input_image = load_input_image(input_image_filepath)
    
    # Resize using Python's resize library with the same parameters as Java
    zoom_factors = (0.5, 0.5)  # Match Java zoom factor
    interpolation = 'Cubic'  # Match Java interpolation method
    python_resized_image = resize_image(input_image, zoom_factors=zoom_factors, interpolation=interpolation)
    
    # Ensure dimensions match between Java and Python results
    assert java_resized_image.shape == python_resized_image.shape, \
        "Java and Python resized images have different dimensions."
    
    # Calculate MSE and PSNR for comparison
    mse, psnr = calculate_metrics(java_resized_image, python_resized_image)
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Peak Signal-to-Noise Ratio (PSNR): {psnr} dB")

if __name__ == "__main__":
    main()
