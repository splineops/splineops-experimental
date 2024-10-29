import numpy as np

def recursive_smoothing_spline(signal, lam=1.0):
    """
    Applies recursive smoothing spline filtering to the input signal using
    a causal and anticausal IIR filter.
    
    Parameters:
    - signal: 1D array of data points to smooth
    - lam: Smoothing parameter controlling the amount of smoothing (lambda)

    Returns:
    - smoothed_signal: 1D array of smoothed data
    """
    # Define the filter pole (z1) based on the regularization parameter lambda
    z1 = -lam / (1 + np.sqrt(1 + 4 * lam))
    K = len(signal)
    
    # Causal filtering (forward pass)
    y_causal = np.zeros(K)
    y_causal[0] = signal[0]
    for k in range(1, K):
        y_causal[k] = signal[k] + z1 * y_causal[k - 1]

    # Anticausal filtering (backward pass)
    smoothed_signal = np.zeros(K)
    smoothed_signal[-1] = y_causal[-1]
    for k in range(K - 2, -1, -1):
        smoothed_signal[k] = y_causal[k] + z1 * smoothed_signal[k + 1]
        
    return smoothed_signal
