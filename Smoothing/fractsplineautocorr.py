import numpy as np

def fractsplineautocorr(alpha, nu):
    """
    Frequency domain computation of fractional spline autocorrelation.

    Parameters:
    alpha (float): The fractional degree parameter.
    nu (array-like): Frequency values.

    Returns:
    A (numpy array): Frequency response of the autocorrelation filter.
    """
    N = 100  # Number of terms in the summation

    if alpha <= -0.5:
        print('The autocorrelation of the fractional splines exists only for degrees strictly larger than -0.5!')
        return []

    S = np.zeros(len(nu))
    for n in range(-N, N + 1):
        S += np.abs(np.sinc(nu + n)) ** (2 * alpha + 2)

    U = 2 / ((2 * alpha + 1) * N ** (2 * alpha + 1))
    U -= 1 / N ** (2 * alpha + 2)
    U += (alpha + 1) * (1 / 3 + 2 * nu ** 2) / N ** (2 * alpha + 3)
    U -= (alpha + 1) * (2 * alpha + 3) * nu ** 2 / N ** (2 * alpha + 4)
    U *= np.abs(np.sin(np.pi * nu) / np.pi) ** (2 * alpha + 2)

    A = S + U
    return A
