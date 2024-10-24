import numpy as np
import matplotlib.pyplot as plt
from fBmper import fBmper
from smoothspline import smoothspline

# Define program constants
m = 4       # Upsampling factor
N = 256     # Number of samples

print('fsdemo.py')
print('--------')
print('Optimal estimation of fractional Brownian motion (fBm)')
print('using fractional splines.\n')
print('Biomedical Imaging Group, EPFL, 2006.\n')

print(f'In this demo, a realization of an fBm process of length {N} is generated and corrupted with noise.')
print(f'The sequence is then denoised and oversampled by a factor of {m} using the optimal fractional spline estimator.\n')

# Default values
default_H = 0.7
default_SNRmeas = 20.0
default_verify = '0'

# Read parameters H, epsH, sigma
while True:
    H_input = input('Enter Hurst parameter [0 < H < 1] (default: 0.7) > ')
    if H_input == '':
        H = default_H
        break
    else:
        try:
            H = float(H_input)
            if 0 < H < 1:
                break
            else:
                print('Please enter a value between 0 and 1.')
        except ValueError:
            print('Invalid input. Please enter a numerical value.')

SNRmeas_input = input(f'Enter measurement SNR at the mid-point (t = {N/2}) [dB] (default: 20.0) > ')
if SNRmeas_input == '':
    SNRmeas = default_SNRmeas
else:
    try:
        SNRmeas = float(SNRmeas_input)
    except ValueError:
        print('Invalid input. Using default value.')
        SNRmeas = default_SNRmeas

# Create pseudo-fBm signal
epsH = 1
t0, y0 = fBmper(epsH, H, m, N)
Ch = epsH ** 2 / (np.math.gamma(2 * H + 1) * np.sin(np.pi * H))
POWmid = Ch * (N / 2) ** (2 * H)  # Theoretical fBm variance at the midpoint

# Measurement: downsample and add noise
t = t0[::m]
y = y0[::m]
sigma = np.sqrt(POWmid) / (10 ** (SNRmeas / 20))
noise = np.random.randn(N)
noise = sigma * noise / np.sqrt(np.mean(noise ** 2))
y_noisy = y + noise

# Find smoothing spline fit
lambda_ = (sigma / epsH) ** 2
gamma_ = H + 0.5
ts, ys = smoothspline(y_noisy, lambda_, m, gamma_)

# Add non-stationary correction term
cnn = np.concatenate(([1], np.zeros(N - 1)))  # Normalized white noise autocorrelation
tes, r = smoothspline(cnn, lambda_, m, gamma_)
r = r * ys[0] / r[0]
y_est = ys - r

# Calculate MSE and SNR
MSE0 = np.mean(noise ** 2)                       # Measurement MSE
MSE = np.mean((y_est[::m] - y0[::m]) ** 2)       # Denoised sequence MSE
MSEm = np.mean((y_est - y0) ** 2)                # Denoised and oversampled signal MSE

SNR0 = 10 * np.log10(POWmid / MSE0)
SNR = 10 * np.log10(POWmid / MSE)
SNRm = 10 * np.log10(POWmid / MSEm)

print('\n')
print(f'Number of measurements is {N}, oversampling factor is {m}.')
print(f'mSNR (SNR at the mid-point) of the measured sequence      is {SNR0:.2f} dB.')
print(f'mSNR improvement of the denoised sequence                 is {SNR - SNR0:.2f} dB.')
print(f'mSNR improvement of the denoised and oversampled sequence is {SNRm - SNR0:.2f} dB.')

# Plot the results
plt.figure()
plt.plot(t0[:len(t0)//2], y0[:len(y0)//2], 'k', label='fBm')
plt.plot(t[:len(t)//2], y_noisy[:len(y_noisy)//2], 'k+:', label='Noisy fBm samples')
plt.plot(ts[:len(ts)//2], ys[:len(ys)//2], 'r--', label='Stationary estimation')
plt.plot(tes[:len(tes)//2], y_est[:len(y_est)//2], 'r', label='Non-stationary estimation')
plt.legend()
plt.title(f'Estimation of fBm (H = {H}, ε_H^2 = {epsH}, σ_N^2 = {sigma ** 2:.4f})')
plt.xlabel('Time')
plt.ylabel('B_H')
plt.tight_layout()
plt.show()

# Verification (Optional)
print('\n')
print('NOTE: To verify the optimality of the estimator we compare the MSE for estimators with different values of gamma and lambda.')
print('To avoid excessive computation, verification is performed using only one realization of fBm.\n')

verify = input('Enter 1 to verify results (may take some time) or 0 to end (default: 0) > ')
if verify == '':
    verify = default_verify

if verify != '1':
    print('\nDone.')
    exit()

# Check for optimality of lambda
Lambda = lambda_ * np.arange(0, 3.1, 0.1)
MSE_list = []
for lam in Lambda:
    tsc, ysc = smoothspline(y_noisy, lam, m, gamma_)
    trc, rc = smoothspline(cnn, lam, m, gamma_)
    y_est_c = ysc - rc * ysc[0] / rc[0]
    MSE_list.append(np.mean((y_est_c[::m] - y0[::m]) ** 2))

plt.figure()
plt.plot(Lambda, MSE_list, 'k', label='MSE vs λ')
plt.plot(lambda_, MSE, 'r+', label='Theoretical optimum point')
plt.legend()
plt.title('MSE vs λ')
plt.xlabel('λ')
plt.ylabel('MSE')
plt.show()

# Check for optimality of gamma_
Gamma = np.arange(0.55, 1.46, 0.01)
MSE_gamma = []
for g in Gamma:
    tsc, ysc = smoothspline(y_noisy, lambda_, m, g)
    trc, rc = smoothspline(cnn, lambda_, m, g)
    y_est_c = ysc - rc * ysc[0] / rc[0]
    MSE_gamma.append(np.mean((y_est_c[::m] - y0[::m]) ** 2))

plt.figure()
plt.plot(Gamma, MSE_gamma, 'k', label='MSE vs γ')
plt.plot(gamma_, MSE, 'r+', label='Theoretical optimum point')
plt.legend()
plt.title('MSE vs γ')
plt.xlabel('γ')
plt.ylabel('MSE')
plt.show()

print('\nDone.')
