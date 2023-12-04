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

relative_errors = []

# Zeitvektor (von 0 bis 30, in 0,01 Sekunden-Schritten)
t = np.arange(0, 30, 0.01)

# Eingangssignal definieren 1, solang wie t
u = np.ones_like(t)

# Koeffizienten bei SysId
na = 1  # Anzahl der Ausgabekoeffizienten
nb = 1  # Anzahl der Eingabekoeffizienten

# ----------------------------------
# FUNCTIONS
# ----------------------------------


# ----------------------------------
# PREPROCESSING
# ----------------------------------

# Zähler und Nenner zusammenfügen zu Übertragungsfunktion
system = co.tf(num, den)


# ----------------------------------
# MAINPROCESSING
# ----------------------------------

# System linear simulieren
y, t_out, x_out = com.lsim(system, u, t)

# Ausgangssignal mit Rauschen überlagern
std_dev = 0.01 * np.std(y)  # Standardabweichung von 1 % des Ausgangswerts
noise = np.random.normal(0, std_dev, y.shape)  # Mittelwert des Rauschens ist 0, Standardabweichung ist das 1 % des max. Ausgangswertes, die Form ist wie die des Vektors y.
y_with_noise = y + noise

# Systemidentifikation durchführen
m = GEKKO()
ypred, p, K = m.sysid(t=t_out, u=u, y=y_with_noise, pred='meas', na=na, nb=nb)

# Berechnung des relativen Fehlers für jedes Wertepaar
for true_y, pred_y in zip(y_with_noise, ypred):
    error = abs(true_y - pred_y) / abs(true_y) * 100
    relative_errors.append(error)

# Berechnung des RMSE aus dem relativen Fehler
relative_errors = np.array(relative_errors)
rmse = np.sqrt(np.mean(relative_errors**2))

# ----------------------------------
# POSTPROCESSING
# ----------------------------------

# Print RMSE
print('RMSE:', rmse)

# Figure erstellen für Diagramme
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plott 1: Systemantwort mit Rauschen
axs[0].plot(t_out, y_with_noise, label='Ausgangssignal mit Rauschen')
axs[0].plot(t_out, y, label='Ausgangssignal', linestyle='--')
axs[0].plot(t_out, ypred, label='Identifiziertes Ausgangssignal')
axs[0].set_xlabel('Zeit [s]')
axs[0].set_ylabel('Systemantwort')
axs[0].set_title('Systemantwort mit gaußverteiltem weißen Rauschen')

# Plot 2: Relativer Fehler
axs[1].plot(t_out, relative_errors, label='Relativer Fehler')
axs[1].set_xlabel('Zeit [s]')
axs[1].set_ylabel('Prozent [%]')
axs[1].set_title('Fehler')

# Allgemeine Plot-Einstellungen
plt.legend()
plt.grid(True)
plt.show()


