"""
INFO
Project 3

DESCRIPTION
Abgabe 3 von MEMS Projekt an der TH Köln

FILE
.../project3_ML.py

ASSOCIATED FILES
.../

AUTHOR(S)
Mark Janitschek
TH Köln / F09 / IPK / CAISA
MEMS
WS 2023/24

DATE
05.01.2024

LAST MODIFIED
05.01.2024

Copyright 2023 - TH Köln
"""

# ----------------------------------
# DEPENDENCIES
# ----------------------------------

import control as co
import control.matlab as com
import numpy as np
import matplotlib.pyplot as plt
from gekko import GEKKO
import math
import pandas as pd

# ----------------------------------
# PARAMETERS
# ----------------------------------

t = np.arange(0, 30, 0.01)
amp = 1
freq = 0.1
std_dev = 0.2
jump_sek = 1

# ----------------------------------
# AUFGABE 1
# ----------------------------------

num = [8.3 * 10 ** (-8)]
den = [5, 1]

system_c = co.tf(num, den, 0.01)
#system_d = com.c2d(system_c, 0.01, method='zoh')


# ----------------------------------
# AUFGABE 2
# ----------------------------------

sinus_signal = amp * np.sin(2 * np.pi * freq * t)
noise = np.random.normal(0, std_dev, sinus_signal.shape)
jump_array = [0] * jump_sek * 100 + [10 ** 8] * (3000 - jump_sek * 100)

u = (sinus_signal + noise) * jump_array

y, t_out, x_out = com.lsim(system_c, u, t)


# ----------------------------------
# AUFGABE 3
# ----------------------------------

matrix = []
nu = ny = 11

for i in range(100 + ny, 3000):
    new_line = [u[i], u[i-1], u[i-2], u[i-3], u[i-4], u[i-5], u[i-6], u[i-7], u[i-8], u[i-9], u[i-10], u[i-11], y[i], y[i-1], y[i-2], y[i-3], y[i-4], y[i-5], y[i-6], y[i-7], y[i-8], y[i-9], y[i-10], y[i-11]]
    matrix.append(new_line)

np.savetxt("matrix_aufgabe3_csv", matrix, delimiter=',')

# ----------------------------------
# POSTPROCESSING
# ----------------------------------

# Grid aktivieren für Plots
plt.rcParams['axes.grid'] = True

# Figure 1 für Aufgabe 1 & 2
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig.suptitle('Aufgabe 1 & 2', fontsize=16)

# Plot 1: Eingangssignal
axs[0].plot(t, u, label='Eingangssignal')
axs[0].set_xlabel('Zeit [s]')
axs[0].set_ylabel('')
axs[0].set_title('Eingang')
axs[0].legend()

# Plot 2: Ausgangssignal
axs[1].plot(t_out, y, label='Ausgangssignal')
axs[1].set_xlabel('Zeit [s]')
axs[1].set_ylabel('')
axs[1].set_title('Ausgang')
axs[1].legend()

# Allgemeine Figure-Einstellungen
plt.tight_layout()

# Diagramme zeigen
plt.show()