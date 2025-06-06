import numpy as np
import matplotlib.pyplot as plt

# Load Real Data
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
    6.67, 3.88, 1.67, 1.13, 1.35, 3.22, 4.97, 5.02, 1.96, 0.16,
    0.18, 0.1, 0.14, 0.09, 0.09, 0.13, 0.39, 1.00, 1.83, 2.16
])

# Time and initial values
time_steps = len(inflation_data)
t = np.arange(time_steps)
C0, I0 = spending_annual[0], inflation_data[0]
C1, I1 = spending_annual[1], inflation_data[1]
R0, R1 = 4, 3

# Define the model
def run_option1(alpha, beta, gamma):
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

# Parameters
alpha1, alpha2 = 0.98, 0.25
beta, beta2 = 0.18, 0.4
gamma, gamma2 = 0.2, 0.2

# Run simulations
C_sim1, I_sim1, R_sim1 = run_option1(alpha1, beta, gamma)
C_sim2, I_sim2, R_sim2 = run_option1(alpha2, beta2, gamma2)

# Plot results
fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# Plot Consumer Spending
axs[0].plot(t[1:], spending_annual, 'ro', label='Real Spending')
axs[0].plot(t, C_sim1, '--', label=f'Option 1')
axs[0].plot(t, C_sim2, '--', label=f'Option 2')
axs[0].set_ylabel('Spending Growth Rate (%)')
axs[0].legend()
axs[0].grid(True)

# Plot Inflation
axs[1].plot(t, inflation_data, 'ro', label='Real Inflation')
axs[1].plot(t, I_sim1, '--', label=f'Option 1')
axs[1].plot(t, I_sim2, '--', label=f'Option 2')
axs[1].set_ylabel('Inflation Rate (%)')
axs[1].legend()
axs[1].grid(True)

# Plot Interest Rate
axs[2].plot(t, rate_annual, 'ro', label='Real Interest Rate')
axs[2].plot(t, R_sim1, '--', label=f'Option 1')
axs[2].plot(t, R_sim2, '--', label=f'Option 2')
axs[2].set_xlabel('Year')
axs[2].set_ylabel('Interest Rate (%)')
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()

