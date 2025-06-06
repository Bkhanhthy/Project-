import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ----------------------------------
# Load Real Data
# ----------------------------------

inflation_data = np.array([
   2.5, 2.7, 2.2, 1.3, 2.0, 2.1, 2.7, 2.3, 2.3, 1.6, 0.8, 2.0, 2.0, 1.7, 1.7, 1.9, 2.2, 1.7, 2.2, 2.3
])

spending_annual_raw = np.array([
    6.7, 7.1, 7.3, 7.7, 8.2, 8.8, 9.3, 9.7, 10, 9.9,
    10.26, 10.699, 11.047, 11.388, 11.874,
    12.297, 12.727, 13.291, 13.934, 14.418
])

spending_annual = 100 * (spending_annual_raw[1:] - spending_annual_raw[:-1]) / spending_annual_raw[:-1]

rate_annual = np.array([
    6.67, 3.88, 1.67, 1.13, 1.35, 3.22, 4.97, 5.02, 1.96, 0.16, 0.18, 0.1, 0.14, 0.09, 0.09, 0.13, 0.39, 1.00, 1.83, 2.16
])
# Initial conditions
C0, I0 = spending_annual[0], inflation_data[0]
C1, I1 = spending_annual[1], inflation_data[1]
R0, R1 = 4,3 #normal interest rate

# Model parameters
alpha = 0.98
beta = 0.18
gamma = 0.2

time_steps = len(inflation_data)
t = np.arange(time_steps)

# Model definition
def model(t, alpha, beta, gamma):
    C = np.zeros(time_steps)
    I = np.zeros(time_steps)
    R = np.zeros(time_steps)
    C[0], I[0], R[0] = C0, I0, R0
    C[1], I[1], R[1] = C1, I1, R1
    for i in range(2, time_steps):
        C[i] = C[i-1] - alpha * (I[i-1] - I[i-2]) * C[i-1]
        I[i] = I[i-1] + beta * (C[i-1] - C[i-2]) * I[i-1]
        R[i] = R[i-1] + gamma * (I[i-1] - I[i-2]) * R[i-1]
    return C, I, R

# Define single-param functions
def model_alpha(t, param):
    return model(t, param, beta, gamma)[0]

def model_beta(t, param):
    return model(t, alpha, param, gamma)[0]

def model_gamma(t, param):
    return model(t, alpha, beta, param)[0]

# Sensitivity analysis function
def localSensitivity_scalar(func, t, a, h=0.01):
    y0 = func(t, (1 - h) * a)
    y1 = func(t, a)
    y2 = func(t, (1 + h) * a)
    return (a / y1) * (y2 - y0) / (2 * h * a)

# Compute sensitivities
S_alpha_scalar = localSensitivity_scalar(model_alpha, t, alpha)
S_beta_scalar = localSensitivity_scalar(model_beta, t, beta)
S_gamma_scalar = localSensitivity_scalar(model_gamma, t, gamma)

# Compute correlations
corr_ab, pval= pearsonr(S_alpha_scalar, S_beta_scalar)
corr_ag, pval= pearsonr(S_alpha_scalar, S_gamma_scalar)
corr_bg, pval= pearsonr(S_beta_scalar, S_gamma_scalar)

R2_ab = corr_ab**2
R2_ag = corr_ag**2
R2_bg = corr_bg**2
print (R2_ab, R2_ag, R2_bg)

# Plot sensitivities
plt.plot(t, S_alpha_scalar, label='S_alpha')
plt.plot(t, S_beta_scalar, label='S_beta')
plt.plot(t, S_gamma_scalar, label='S_gamma')
plt.xlabel('Time (years)')
plt.ylabel('Sensitivity')
plt.title('Sensitivity over Time')
plt.legend()
plt.grid(True)
plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(S_alpha_scalar, S_beta_scalar)
plt.xlabel('S_a')
plt.ylabel('S_b')
plt.title(f"R² = {R2_ab:.2f}")

plt.subplot(1, 3, 2)
plt.plot(S_alpha_scalar, S_gamma_scalar)
plt.xlabel('S_r')
plt.ylabel('S_a')
plt.title(f"R² = {R2_ag:.2f}")

plt.subplot(1, 3, 3)
plt.plot(S_beta_scalar, S_gamma_scalar)
plt.xlabel('S_r')
plt.ylabel('S_a')
plt.title(f"R² = {R2_bg:.2f}")

plt.tight_layout()
plt.show()

