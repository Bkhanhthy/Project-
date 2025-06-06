#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  9 10:46:02 2025

@author: caylabovell
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ----------------------------------
# Data and Initial Conditions
# ----------------------------------

inflation_data = np.array([
   2.5, 2.7, 2.2, 1.3, 2.0, 2.1, 2.7, 2.3, 2.3, 1.6, 0.8, 2.0, 2.0, 1.7, 1.7, 1.9, 2.2, 1.7, 2.2, 2.3
])

spending_annual_raw = np.array([6.7, 7.1, 7.3, 7.7, 8.2, 8.8, 9.3, 9.7, 10, 9.9,
    10.26, 10.699, 11.047, 11.388, 11.874, 12.297, 12.727, 13.291, 13.934, 14.418
])
spending_annual = 100 * (spending_annual_raw[1:] - spending_annual_raw[:-1]) / spending_annual_raw[:-1]

rate_annual = np.array([
    6.67, 3.88, 1.67, 1.13, 1.35, 3.22, 4.97, 5.02, 1.96, 0.16, 0.18, 0.1, 0.14, 0.09, 0.09, 0.13, 0.39, 1.00, 1.83, 2.16
])

time_steps = len(inflation_data)
C0, I0, R0 = 3,2,6
C1, I1, R1 = 5, 5, 3.78

# ----------------------------------
# Model Function (no max)
# ----------------------------------
def run_option1(alpha, beta, gamma):
    C = np.zeros(time_steps)
    I = np.zeros(time_steps)
    R = np.zeros(time_steps)
    C[0], I[0], R[0] = C0, I0, R0
    C[1], I[1], R[1] = C1, I1, R1
    for i in range(2, time_steps):
        C[i] = C[i-1] - alpha * (I[i-1] - I[i-2]) * C[i-1]
        I[i] = I[i-1] + beta  * (C[i-1] - C[i-2]) * I[i-1]
        R[i] = R[i-1] + gamma * (I[i-1] - I[i-2]) * R[i-1]
    return C, I, R

# Initial parameter values
alpha0, beta0, gamma0 = 0.0002, 0.03, 0.2

# Initial simulation
C_sim, I_sim, R_sim = run_option1(alpha0, beta0, gamma0)
sim_data = [C_sim, I_sim, R_sim]
real_data = [spending_annual, inflation_data, rate_annual]
labels = [
    ('Model Spending', 'Real Spending', 'Spending Rate (%)'),
    ('Model Inflation', 'Real Inflation', 'Inflation (%)'),
    ('Model Interest', 'Real Interest', 'Interest Rate (%)')
]

# ----------------------------------
# Figure Creation Function
# ----------------------------------
def create_figure(idx):
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    line_model, = ax.plot(sim_data[idx], label=labels[idx][0])
    ax.plot(real_data[idx], 'o', label=labels[idx][1])
    ax.set_ylabel(labels[idx][2])
    ax.legend()
    ax.set_xlabel('Time Step')
    plt.subplots_adjust(bottom=0.25)

    # Slider axes for each figure
    ax_alpha = fig.add_axes([0.15, 0.15, 0.7, 0.03])
    ax_beta  = fig.add_axes([0.15, 0.10, 0.7, 0.03])
    ax_gamma = fig.add_axes([0.15, 0.05, 0.7, 0.03])

    slider_alpha = Slider(ax_alpha, 'alpha', 0.0, 1.0, valinit=alpha0)
    slider_beta  = Slider(ax_beta,  'beta',  0.0, 1.0, valinit=beta0)
    slider_gamma = Slider(ax_gamma, 'gamma', 0.0, 1.0, valinit=gamma0)

    def update(val):
        a = slider_alpha.val
        b = slider_beta.val
        g = slider_gamma.val
        C_new, I_new, R_new = run_option1(a, b, g)
        new_data = [C_new, I_new, R_new]
        line_model.set_ydata(new_data[idx])
        fig.canvas.draw_idle()

    slider_alpha.on_changed(update)
    slider_beta.on_changed(update)
    slider_gamma.on_changed(update)

    return fig

# Create three separate figures
fig1 = create_figure(0)
'fig2 = create_figure(1)'
'fig3 = create_figure(2)'

plt.show()