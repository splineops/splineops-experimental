import numpy as np
import matplotlib.pyplot as plt
from smoothspline_recursive import recursive_smoothing_spline
from smoothspline import smoothspline  # Import smoothspline from smoothspline.py

# Example signal: A noisy sine wave
x = np.linspace(0, 2 * np.pi, 100)
signal = np.sin(x) + 0.1 * np.random.normal(size=x.shape)

# Different values for the smoothing parameter in recursive smoothing spline
lam_values = [0.005, 0.05, 0.1]  # You can try smaller or larger values

# Apply fractional smoothing spline as a baseline for comparison
lambda_ = 0.1  # Regularization parameter for fractional method
m = 1          # No upsampling
gamma = 0.6    # Spline order parameter
_, smoothed_fractional = smoothspline(signal, lambda_, m, gamma)

# Plot results
plt.figure(figsize=(12, 8))
plt.plot(x, signal, label="Noisy Signal", linestyle="--", color="gray")
plt.plot(x, smoothed_fractional, label="Fractional Smoothing Spline", color="red")

# Apply and plot recursive smoothing spline for each lambda value
for lam_recursive in lam_values:
    smoothed_recursive = recursive_smoothing_spline(signal, lam=lam_recursive)
    plt.plot(x, smoothed_recursive, label=f"Recursive Smoothing (λ={lam_recursive})")

plt.legend()
plt.xlabel("x")
plt.ylabel("Signal Value")
plt.title("Comparison of Recursive Smoothing with Different λ Values")
plt.show()
