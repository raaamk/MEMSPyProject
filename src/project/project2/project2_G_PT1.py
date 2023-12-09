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
import math
import pandas as pd

# ----------------------------------
# PARAMETERS
# ----------------------------------

# Definieren Sie die Übertragungsfunktion
num = [8.3 * 10 ** (-8)]
den = [5, 1]

n = 3000

i = 0
absoluter_Fehler = np.zeros(n)
relativer_Fehler = np.zeros(n)
RMSE_Zaehler = 0

# Zeitvektor (von 0 bis 30, in 0,01 Sekunden-Schritten)
t = np.arange(0, 30, 0.01)

# Eingangssignal definieren 1, solang wie t
u = np.ones_like(t)

# Koeffizienten bei SysId
na = 1  # Anzahl der Ausgabekoeffizienten
nb = 1  # Anzahl der Eingabekoeffizienten

# Daten einlesen und Matrix erstellen
df = pd.read_csv('mems_identification_data.csv', sep=',', header=0)
daten = np.array(df)
M = np.array([(-0, -0, -0, 0, 0),
              (-daten[0, 1], -0, -0, daten[0, 0], 0), (-daten[1, 1], -daten[0, 1], -0, daten[1, 0], daten[0, 0]), (-daten[2, 1], -daten[1, 1], -daten[0, 1], daten[2, 0], daten[1, 0]), (-daten[3, 1], -daten[2, 1], -daten[1, 1], daten[3, 0], daten[2, 0]),
              (-daten[4, 1], -daten[3, 1], -daten[2, 1], daten[4, 0], daten[3, 0]), (-daten[5, 1], -daten[4, 1], -daten[3, 1], daten[5, 0], daten[4, 0]), (-daten[6, 1], -daten[5, 1], -daten[4, 1], daten[6, 0], daten[5, 0]), (-daten[7, 1], -daten[6, 1], -daten[5, 1], daten[7, 0], daten[6, 0]),
              (-daten[8, 1], -daten[7, 1], -daten[6, 1], daten[8, 0], daten[7, 0]), (-daten[9, 1], -daten[8, 1], -daten[7, 1], daten[9, 0], daten[8, 0]), (-daten[10, 1], -daten[9, 1], -daten[8, 1], daten[10, 0], daten[9, 0]), (-daten[11, 1], -daten[10, 1], -daten[9, 1], daten[11, 0], daten[10, 0])])
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
std_dev = 0.01 * y  # Standardabweichung von 1 % des Ausgangswerts
noise = np.random.normal(0, std_dev, y.shape)  # Mittelwert des Rauschens ist 0, Standardabweichung ist das 1 % des max. Ausgangswertes, die Form ist wie die des Vektors y.
y_with_noise = y + noise

# Systemidentifikation durchführen
m = GEKKO()
ypred, p, K = m.sysid(t=t_out, u=u, y=y_with_noise, pred='meas', na=na, nb=nb)

#
while i < n:
    absoluter_Fehler[i] = abs(y[i] - ypred[i])
    if y[i] != 0:
        relativer_Fehler[i] = absoluter_Fehler[i] / y[i]
    RMSE_Zaehler = RMSE_Zaehler + absoluter_Fehler[i] ** 2
    i = i + 1

rmse = math.sqrt(RMSE_Zaehler / n)

# Aufgabe 5
a1, a2, a3, b1, b2 = np.linalg.lstsq(M, df['y'], rcond=None)[0]

num_data = np.array([b1, b2])
den_data = np.array([a1, a2, a3])

G = co.tf(num_data, den_data)

# ----------------------------------
# POSTPROCESSING
# ----------------------------------

# Print
print('RMSE:', rmse)
print(G)

# Figure erstellen für Diagramme
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Plott 1: Systemantwort mit Rauschen
axs[0].plot(t_out, y_with_noise, label='Ausgangssignal mit Rauschen')
axs[0].plot(t_out, y, label='Ausgangssignal', linestyle='--')
axs[0].plot(t_out, ypred, label='Identifiziertes Ausgangssignal')
axs[0].set_xlabel('Zeit [s]')
axs[0].set_ylabel('Systemantwort')
axs[0].set_title('Systemantwort mit gaußverteiltem weißen Rauschen')
axs[0].legend()

# Plot 2: Relativer Fehler
axs[1].plot(t_out, relativer_Fehler, label='Relativer Fehler')
axs[1].set_xlabel('Zeit [s]')
axs[1].set_ylabel('Prozent [%]')
axs[1].set_title('Relativer Fehler')
axs[1].legend()

# Plot 3: Absoluter Fehler
axs[2].plot(t_out, absoluter_Fehler, label='Absoluter Fehler')
axs[2].set_xlabel('Zeit [s]')
axs[2].set_ylabel('Prozent [%]')
axs[2].set_title('Absoluter Fehler')
axs[2].legend()

# Allgemeine Plot-Einstellungen
plt.grid(True)
plt.tight_layout()
plt.show()
