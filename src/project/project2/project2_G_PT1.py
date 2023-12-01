"""
INFO
Project 2

INPUT VARIABLES
-

OUTPUT VARIABLES
-

DESCRIPTION
-

FILE
.../project2_G_PT1.py

ASSOCIATED FILES
-

AUTHOR(S)
Mark Janitschek
TH Köln / F09 / IPK / CAISA
MEMS
WS 2023/24

DATE
29.11.2023

LAST MODIFIED
29.11.2023

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

# ----------------------------------
# PARAMETERS
# ----------------------------------

# Definieren Sie die Übertragungsfunktion
num = [8.3 * 10 ** (-8)]
den = [5, 1]
system = co.tf(num, den)

# Zeitvektor (von 0 bis 30, in 0,01 Sekunden-Schritten)
t = np.arange(0, 30, 0.01)

# Eingangssignal definieren 1, solang wie t
u = np.ones_like(t)

# ----------------------------------
# FUNCTIONS
# ----------------------------------


# ----------------------------------
# PREPROCESSING
# ----------------------------------


# ----------------------------------
# MAINPROCESSING
# ----------------------------------

# Main
y, t_out, x_out = com.lsim(system, u, t)

# Ausgangssignal mit Rauschen überlagern
std_dev = 0.01 * np.std(y)  # Standardabweichung von 1 % des Ausgangswerts
noise = np.random.normal(0, std_dev, y.shape)  # Mittelwert des Rauschens ist 0, Standardabweichung ist das 1 % des max. Ausgangswertes, die Form ist wie die des Vektors y.
y_with_noise = y + noise

# Systemidentifikation durchführen
m = GEKKO()
ypred, p, K = m.sysid(t=t_out, u=u, y=y_with_noise, pred='meas')

# ----------------------------------
# POSTPROCESSING
# ----------------------------------

# Plotten Sie die Systemantwort mit Rauschen
plt.plot(t_out, y_with_noise, label='Ausgangssignal mit Rauschen')
plt.plot(t_out, y, label='Ausgangssignal', linestyle='--')
plt.plot(t_out, ypred, label='Identifiziertes Ausgangssignal')
plt.xlabel('Zeit')
plt.ylabel('Systemantwort')
plt.title('Systemantwort mit gaußverteiltem weißen Rauschen')
plt.legend()
plt.grid(True)
plt.show()
