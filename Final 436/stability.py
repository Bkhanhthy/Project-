#Stabilty Analysis 

import numpy as np
import matplotlib.pyplot as plt

# Equilibrium values (policy targets)
C_eq = 2.0  # Spending growth (%)
I_eq = 2.0  # Inflation (%)
R_eq = 1.5  # Interest rate (%)

# Model parameters
alpha = 0.98
beta = 0.18
gamma = 0.2

# Simulation setup
steps = 30
delta_C = np.zeros(steps)
delta_I = np.zeros(steps)
delta_R = np.zeros(steps)

# Initial deviations from equilibrium
delta_C[0], delta_C[1] = 0.01, 0.0
delta_I[0], delta_I[1] = 0.01, 0.0
delta_R[0], delta_R[1] = 0.01, 0.0

# Linearized perturbation system
for i in range(2, steps):
    delta_C[i] = delta_C[i-1] - alpha * C_eq * (delta_I[i-1] - delta_I[i-2])
    delta_I[i] = delta_I[i-1] + beta * I_eq * (delta_C[i-1] - delta_C[i-2])
    delta_R[i] = delta_R[i-1] + gamma * R_eq * (delta_I[i-1] - delta_I[i-2])

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(delta_C, label='ΔC (Spending Deviation)')
plt.plot(delta_I, label='ΔI (Inflation Deviation)')
plt.plot(delta_R, label='ΔR (Interest Rate Deviation)')
plt.axhline(0, color='gray', linestyle='--')
plt.title('Stability Analysis: Deviations from Equilibrium Over Time')
plt.xlabel('Time(Years)')
plt.ylabel('Deviation from Equilibrium')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#Jacobian & eigenvalues
import numpy as np
import pandas as pd

# Parameters and equilibrium
alpha, beta, gamma = 0.98, 0.18, 0.2
C_eq, I_eq, R_eq = 2.0, 2.0, 1.5

# Build the 6×6 Jacobian
J = np.zeros((6, 6))
J[0, 1] = 1
J[0, 3] = -alpha * C_eq
J[0, 2] =  alpha * C_eq
J[1, 0] =  1
J[2, 3] =  1
J[2, 1] =  beta * I_eq
J[2, 0] = -beta * I_eq
J[3, 2] =  1
J[4, 5] =  1
J[4, 3] =  gamma * R_eq
J[4, 2] = -gamma * R_eq
J[5, 4] =  1

# Compute eigenvalues
eigvals = np.linalg.eigvals(J)

# Display
print("Jacobian matrix J:")
print(J)

df = pd.DataFrame({
    'Eigenvalue': eigvals,
    'Magnitude': np.abs(eigvals)
})
print("\nEigenvalues and their magnitudes:")
print(df.to_markdown(index=False))