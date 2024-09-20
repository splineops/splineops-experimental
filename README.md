# Least-Squares Image Resizing Using Finite Differences

This repository contains a Python implementation of the algorithm described in the paper [Least-Squares Image Resizing Using Finite Differences](https://bigwww.epfl.ch/publications/munoz0101.html). The algorithm provides an optimal spline-based approach for resizing digital images with arbitrary (non-integer) scaling factors. It minimizes artifacts such as aliasing and blocking, improving the signal-to-noise ratio compared to standard interpolation methods.

## Features
- Optimal spline-based image resizing
- Arbitrary scaling factors
- Consistent reduction of artifacts and improved signal-to-noise ratio
- Computational complexity per pixel is independent of the scaling factor

## Repository Structure
- `Resize/`
  - `Resize.py`: Core library implementing the resizing algorithm.
  - `Original_Source_Java_Code/`: Directory containing the original Java implementation of the algorithm.
  - `Samples/`: Directory containing sample images for testing.
  - `Results/`: Directory containing plots and result images from the comparison tests.
  - `testComparison_LS_Arrate_Ground_Truth.py`: Comparing the code results with the original Java source.
  - `testComparison_LS_Interp_Oblique_Plot.py`: Compares LS, basic interpolation and Oblique interpolation across several scaling factors.
  - `testComparison_LS_Interp_Oblique.py`: Compares LS, basic interpolation and Oblique interpolation.
  - `testComparison_LS_Linear_Cubic_Plot.py`: Compares LS between Linear and Cubic interpolation.
  - `testComparison_LS_Oblique_Scipy_Plot.py`: Compares Oblique interpolation and Scipy interpolation across several scaling factors.
  - `testComparison_LS_Scipy_Plot.py`: Compares LS and Scipy interpolation across several scaling factors.
  - `testComparison_LS_Scipy.py`: Compares LS and Scipy interpolation.

## Installation and Usage
To use the core library, only `python` and `numpy` is required. To run the test examples, you will need additional dependencies.

1. Clone the repository:
    ```bash
    git clone https://github.com/splineops/splineops-experimental.git
    cd splineops-experimental/Resize
    ```

2. Install the required dependencies:
    ```bash
    pip install numpy scipy matplotlib imageio
    ```

3. To run the tests, just execute them with Python.