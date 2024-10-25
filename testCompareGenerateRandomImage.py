import numpy as np
import csv

# Generate a 500x500 random image and save it to CSV
def generate_and_save_image(filepath, width=500, height=500):
    np.random.seed(0)  # Optional: set seed for reproducibility
    random_image = np.random.rand(height, width)
    
    # Save to CSV
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Input Image Data"])
        for row in random_image:
            writer.writerow(row)

generate_and_save_image("input_image.csv")
