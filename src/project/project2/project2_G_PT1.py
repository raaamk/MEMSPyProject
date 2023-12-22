"""
INFO
Project 2

DESCRIPTION
Abgabe 2 von MEMS Projekt an der TH Köln

FILE
.../project2_G_PT1.py

ASSOCIATED FILES
.../mems_identification_data.csv

AUTHOR(S)
Mark Janitschek
TH Köln / F09 / IPK / CAISA
MEMS
WS 2023/24

DATE
29.11.2023

LAST MODIFIED
22.12.2023

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

# Definieren der Übertragungsfunktion für Aufgabe 1, 2 & 3
num = [8.3 * 10 ** (-8)]
den = [5, 1]

# Definieren der Übertragungsfunktion für Aufgabe 4 & 5
num2 = [3]
den2 = [6, 2, 1]

# Zeitvektoren
t = np.arange(0, 30, 0.01)  # von 0 bis 30, in 0,01 Sekunden-Schritten
t2 = np.arange(0, 41.5, 0.1)  # von 0 bis 41.5, in 0,1 Sekunden-Schritten

# Eingangssignal definieren 1, solang wie t
u = np.ones_like(t)

# Variablen für Aufgabe 1, 2 & 3
n = len(t)  # Anzahl der Datenpunkte von Zeitvektor
i = 0
absoluter_Fehler = np.zeros(n)
relativer_Fehler = np.zeros(n)
RMSE_Zaehler = 0

# Variablen für Aufgabe 4 & 5
n2 = len(t2)
absoluter_Fehler2 = np.zeros(n2)
relativer_Fehler2 = np.zeros(n2)
RMSE_Zaehler2 = 0

# Koeffizienten bei SysId
na = 1  # Anzahl der Ausgabekoeffizienten
nb = 1  # Anzahl der Eingabekoeffizienten

# Daten einlesen und Matrix erstellen
daten_pd = pd.read_csv('mems_identification_data.csv', sep=',', header=0)
daten = np.array(daten_pd)
y_data_11 = daten[-11:, 1]  # Letzten 11 Werte von y in Array
M = np.array([(-daten[1, 1], -daten[0, 1], daten[1, 0], daten[0, 0]),
              (-daten[2, 1], -daten[1, 1], daten[2, 0], daten[1, 0]),
              (-daten[3, 1], -daten[2, 1], daten[3, 0], daten[2, 0]),
              (-daten[4, 1], -daten[3, 1], daten[4, 0], daten[3, 0]),
              (-daten[5, 1], -daten[4, 1], daten[5, 0], daten[4, 0]),
              (-daten[6, 1], -daten[5, 1], daten[6, 0], daten[5, 0]),
              (-daten[7, 1], -daten[6, 1], daten[7, 0], daten[6, 0]),
              (-daten[8, 1], -daten[7, 1], daten[8, 0], daten[7, 0]),
              (-daten[9, 1], -daten[8, 1], daten[9, 0], daten[8, 0]),
              (-daten[10, 1], -daten[9, 1], daten[10, 0], daten[9, 0]),
              (-daten[11, 1], -daten[10, 1], daten[11, 0], daten[10, 0])])


# ----------------------------------
# AUFGABE 1, 2 & 3
# ----------------------------------

# Zähler und Nenner zusammenfügen zu Übertragungsfunktion
system = co.tf(num, den)

# System linear simulieren
y, t_out, x_out = com.lsim(system, u, t)

# Ausgangssignal mit Rauschen überlagern
std_dev = 0.01 * y  # Standardabweichung von 1 % des Ausgangswerts
noise = np.random.normal(0, std_dev, y.shape)  # Mittelwert des Rauschens ist 0, Standardabweichung ist das 1 % des max. Ausgangswertes, die Form ist wie die des Vektors y.
y_with_noise = y + noise

# Systemidentifikation durchführen
m = GEKKO()
ypred, p, K = m.sysid(t=t_out, u=u, y=y_with_noise, pred='meas', na=na, nb=nb)

# Relativer & absoluter Fehler und RMSE berechnen für Aufgabe 3
while i < n:
    absoluter_Fehler[i] = y[i] - ypred[i]
    if y[i] != 0:
        relativer_Fehler[i] = absoluter_Fehler[i] / y[i]
    RMSE_Zaehler = RMSE_Zaehler + absoluter_Fehler[i] ** 2
    i = i + 1

rmse = math.sqrt(RMSE_Zaehler / n)

# ----------------------------------
# AUFGABE 4 & 5
# ----------------------------------

# Parameter bestimmen
a1, a2, b1, b2 = np.linalg.lstsq(M, y_data_11, rcond=None)[0]

# Berechnete Parameter in Arrays
num_data = np.array([b1, b2, 0])
den_data = np.array([1, a1, a2])

# Übertragungsfunktion erstellen
G_data = co.tf(num_data, den_data, 0.1)  # aus den Daten mit Zeitkonstante 0.01
system2 = co.tf(num2, den2)  # von System 2

# System 2 von kontinuierlich in zeitdiskret umgewandelt
system2_c2d = com.c2d(system2, 0.1, method='zoh')

# Systeme mit Sprungantwort simulieren
t_out_data, y_data = co.step_response(G_data)  # von Daten identifiziertes System
t_out_2, y_2 = co.step_response(system2_c2d)  # von System 2, das im zeitdiskret umgewandelt wurde

# Relativer & absoluter Fehler und RMSE berechnen für Aufgabe 5
for j in range(0, n2):
    absoluter_Fehler2[j] = (y_2[j] - y_data[j])
    if y_data[j] != 0:
        relativer_Fehler2[j] = absoluter_Fehler2[j] / y_data[j]
    RMSE_Zaehler2 = RMSE_Zaehler2 + absoluter_Fehler2[j] ** 2

rmse2 = math.sqrt(RMSE_Zaehler2 / len(y_data))

# ----------------------------------
# POSTPROCESSING
# ----------------------------------

# Print
print('-------------------------------------------------')
print('RMSE aus Aufgabe 3:', rmse)
print('-------------------------------------------------')
print('Übertragungsfunktion aus eingelesenen Daten:', G_data)
print('-------------------------------------------------')
print('RMSE aus Aufgabe 5:', rmse2)

# Grid aktivieren für Plots
plt.rcParams['axes.grid'] = True

# Figure 1 für Aufgabe 1, 2 & 3 erstellen für Diagramme
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fig.suptitle('Aufgabe 1, 2 & 3', fontsize=16)

# Plott 1: Systemantwort mit Rauschen von Aufgabe 1, 2 % 3
axs[0].plot(t_out, y_with_noise, label='Ausgangssignal mit Rauschen')
axs[0].plot(t_out, y, label='Ausgangssignal', linestyle='--')
axs[0].plot(t_out, ypred, label='Identifiziertes Ausgangssignal')
axs[0].set_xlabel('Zeit [s]')
axs[0].set_ylabel('Systemantwort')
axs[0].set_title('Systemantwort mit gaußverteiltem weißen Rauschen (Aufgabe 1, 2 & 3)')
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
axs[2].set_ylabel('Fehler')
axs[2].set_title('Absoluter Fehler')
axs[2].legend()

# Allgemeine Figure-Einstellungen
plt.tight_layout()

# Figure 2 für Aufgabe 4 & 5 erstellen für Diagramme
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fig.suptitle('Aufgabe 4 & 5', fontsize=16)

# Plot 1: Systemantwort von Aufgabe 4 & 5
axs[0].plot(t_out_data, y_data, label='Ausgangssignal vom identifizierten System')
axs[0].plot(t_out_2, y_2, label='Ausgangssignal von System 2')
axs[0].set_xlabel('Zeit [s]')
axs[0].set_ylabel('Systemantwort')
axs[0].set_title('Systemantwort')
axs[0].legend()

# Plot 2: Relativer Fehler
axs[1].plot(t2, relativer_Fehler2, label='Relativer Fehler')
axs[1].set_xlabel('Zeit [s]')
axs[1].set_ylabel('Prozent [%]')
axs[1].set_title('Relativer Fehler')
axs[1].legend()

# Plot 2: Absoluter Fehler
axs[2].plot(t2, absoluter_Fehler2, label='Absoluter Fehler')
axs[2].set_xlabel('Zeit [s]')
axs[2].set_ylabel('Fehler')
axs[2].set_title('Absoluter Fehler')
axs[2].legend()

# Allgemeine Figure-Einstellungen
plt.tight_layout()

# Diagramme zeigen
plt.show()
