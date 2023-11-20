'''
INFO
This is a template of script.

SYNTAX
-

INPUT VARIABLES
-

OUTPUT VARIABLES
-

DESCRIPTION
This template has predefined sections. Use those that meet your criteria. Add further ones if required.
Substitute comments with meaningful ones assigned to the specific tasks.

SEE ALSO
-

FILE
.../project_windturbine.py

ASSOCIATED FILES
-

AUTHOR(S)
Mark Janitschek
TH Köln / F09 / IPK / CAISA
MEMS
WS 2023/24

DATE
06.11.2023

LAST MODIFIED
-

Copyright 2023 - TH Köln
'''
# ----------------------------------
# DEPENDENCIES
# ----------------------------------

import math
import matplotlib.pyplot as plt
import numpy as np
from WeatherDataFetcher import WeatherDataFetcher

# ----------------------------------
# PARAMETERS
# ----------------------------------

# Umgebung
rho_air = 1.225  # Dichte der Luft
la = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 50.0]  # Leistungsbeiwert-Schnelllaufzahl-Charakteristik
c_p = [0.4, 0.45, 0.48, 0.46, 0.44, 0.4, 0.33, 0.0]
K_m = 75.648  # Generatorkonstante

# Geometrie
l_R = 40  # Länge eines Rotorblattes

# Generatorstrang
n_G = 0.97  # Wirkungsgrad des Generators
J_0 = 3.016 * 10 ** 4  # Massenträgheitsmoment der Antriebswelle
J_1 = 1.624 * 10 ** 4  # Massenträgheitsmoment der Abtriebswelle
b_0 = 5  # Reibbeiwert der Antriebswelle
b_1 = 4  # Reibbeiwert der Abtriebswelle
i_G1 = 1 / 100  # Übersetzung Getriebe

# Windnachführung
n_M = 0.95   # Wirkungsgrad des Motors
J_G = 2.1 * 10 ** 8  # Massenträgheitsmoment der Gondel
b_G = 2.2 * 10 ** 4  # Reibbeiwert der Gondellagerung
i_G2 = 1000  # Übersetzung (ins Langsame)

# Input
T = 0.01  # Zeit/Abtastrate
iteration = 300000  # Anzahl Iterationen
M_B = 0  # Bremsmoment
M_G = 0  # Antriebsmoment Gondel

# Output
w = [0]  # Winkelgeschwindigkeit Antriebsstrang
w_ab = [0]  # Winkelgeschwindigkeit Abtriebsstrang
w_G = [0]  # Winkelgeschwindigkeit Gondel
alpha_G_rad = [0]  # Winkel der Gondel in rad
alpha_G_deg = [0]  # Winkel der Gondel in degree
alpha_G_deg_plot = [0] # Winkel der Gondel in degree immer zwischen 0 und 360 Grad für Plot
iteration_time = [0]  # Zei der Iterationen
delta = [0]  # Winkel zwischen Gondelausrichtung und Windrichtung
P_w = [0]  # Windleistung
P_M = [0]  # Mechanische Leistung des Generators
P_E = [0]  # Elektrische Leistung des Generators
P_G = [0]  # Antriebsleistung Gondel


# ----------------------------------
# FUNCTIONS
# ----------------------------------


# ----------------------------------
# PREPROCESSING
# ----------------------------------

# Aktuelle Wetterdaten holen
weatherdata = WeatherDataFetcher()
weatherdata.save_weather_data()
v_w = weatherdata.saved_windspeed
w_d = weatherdata.saved_winddirection  # in [rad]

# Eigene Werte zum Testen eintippen
#v_w = 11  # Möglich eigenen Wert einzutippen, sonst auskommentieren
#w_d = math.radians(310)  # Möglich eigenen Wert einzutippen, sonst auskommentieren
# ----------------------------------
# MAINPROCESSING
# ----------------------------------

# Execute main operation
for i in range(1, iteration + 1):
    delta_current = (math.degrees(w_d) - alpha_G_deg[i-1] + 180) % 360 - 180  # Bringt die Differenz auf einen Wert zwischen -180 und 180

    v_w_R = v_w * math.cos(math.radians(delta_current))  # Berechnet die auf das Rotorblatt wirkende Windgeschwindigkeit (math.cos rechnet mit radians)

    la_calc = l_R * w[i - 1] / v_w_R  # Berechnet Lambda

    # Überprüfung, ob Lambda im Bereich ist
    if la_calc < 5:
        la_calc = 5.0
    elif la_calc > 50:
        la_calc = 50.0

    c_p_interp = np.interp(la_calc, la, c_p)  # interpoliert den Wert für cp, mithilfe der Leistungsbeiwert-Schnelllaufzahl-Charakteristik und dem berechneten Lambda
    c_m = c_p_interp / la_calc

    # Berechnet die aktuelle Winkelgeschwindigkeit des Antriebsstrangs [rad/s]
    w.append((T / (J_0 + (J_1 / i_G1 ** 2)) * (c_m * 0.5 * rho_air * math.pi * l_R ** 3 * v_w_R ** 2 - w[i - 1] * (b_0 + (b_1 / i_G1 ** 2) + (K_m / i_G1 ** 2)) - M_B)) + w[i - 1])

    # Berechnet die Winkelgeschwindigkeit der Gondel [rad/s]
    w_G.append((M_G * 1000 * T) / J_G - (b_G * w_G[i - 1] * T) / J_G + w_G[i - 1])
    alpha_G_rad.append(w_G[i] * T + alpha_G_rad[i - 1])  # Berechnet den Winkel der Gondel [rad]
    alpha_G_deg.append(math.degrees(alpha_G_rad[i]))  # Berechnet den Winkel der Gondel [°]
    alpha_G_deg_plot.append(alpha_G_deg[i] % 360)

    # Leistungen berechnen für Generatorstrang
    P_w.append(0.5 * rho_air * math.pi * l_R ** 2 * v_w_R ** 3)  # Berechnet die Windleistung
    w_ab.append(w[i]/i_G1)  # Berechnet die Winkelgeschwindigkeit des Abtriebsstranges [rad/s]
    P_M.append(K_m * (w_ab[i]) * (w_ab[i]))  # Berechnet die mechanische Leistung des Generators
    P_E.append(n_G * P_M[i])  # Berechnet die elektrische Leistung des Generators

    # Wenn der Winkel zwischen Gondel und Windrichtung >20 oder <-20 ist, wird das Antriebsmoment für die Gondel auf den entsprechenden Wert gesetzt.
    delta.append(delta_current)
    if delta_current > 20:
        M_G = 1
    elif delta_current < -20:
        M_G = -1
    else:
        M_G = 0

    # Leistung für Gondel
    P_G.append(w_G[i] * M_G * n_M)  # Berechne die Antriebsleistung für die Gondel

    # Für Erstellung der Plots
    iteration_time.append(T * i)

print('Der Wind kommt aus', math.degrees(w_d), '° und ist', v_w, 'm/s schnell.')
print('Das Windrad ist nun Richtung', (alpha_G_deg_plot[iteration]), '° gerichtet')
print('Der am Windrad ankommende Wind ist', v_w_R, 'm/s schnell')
print('Die elektrische Leistung des Generators (PE) am Zeitpunkt', iteration, '[s] ist', P_E[i], '[W]')
print('Die Winkelgeschwindigkeit des Antriebsstranges ist am Zeitpunkt', iteration, '[s] ist', w[i], '[rad/s]')

# ----------------------------------
# POSTPROCESSING
# ----------------------------------

# Figure erstellen für Diagramme
fig, axs = plt.subplots(3, 1, figsize=(10, 8))

# Plot 1: Winkelgeschwindigkeiten
axs[0].plot(iteration_time, w, label='Antriebswelle w_0')
axs[0].plot(iteration_time, w_G, label='Gondel w_G')
axs[0].set_title("Winkelgeschwindigkeiten")
axs[0].set_xlabel("Zeit in Sekunden")
axs[0].set_ylabel("Winkelgeschwindigkeit [rad/s]")
axs[0].legend()

# Plot 2: Leistungen
axs[1].plot(iteration_time, P_w, label='P_w')
axs[1].plot(iteration_time, P_M, label='P_M')
axs[1].plot(iteration_time, P_E, label='P_E')
axs[1].plot(iteration_time, P_G, label='P_G')
axs[1].set_title("Leistungen")
axs[1].set_xlabel("Zeit in Sekunden")
axs[1].set_ylabel("Leistung [W]")
axs[1].legend()

# Plot 3: Wind- und Gondelausrichtung
axs[2].plot(iteration_time, [math.degrees(w_d)]*len(iteration_time), label='Windrichtung')
axs[2].plot(iteration_time, alpha_G_deg_plot, label='Gondelrichtung')
axs[2].set_title("Wind- und Gondelausrichtung")
axs[2].set_xlabel("Zeit in Sekunden")
axs[2].set_ylabel("Richtung [°]")
axs[2].set_yticks(np.arange(0, 400, 40))
axs[2].legend()

# Einstellungen für das gesamte Diagramm
plt.tight_layout()
plt.show()