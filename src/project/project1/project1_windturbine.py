"""
INFO
Project Windkraftanlage

INPUT VARIABLES
iteration, T, v_w, w_d, get_weather

OUTPUT VARIABLES
w, w_ab, w_G, alpha_G_rad, alpha_G_deg, alpha_G_deg_plot, iteration_time, delta, P_W, P_M, P_E, P_G

DESCRIPTION
Dieses Programm modelliert eine Windkraftanlage mit Generatorstrang und Windnachführung.

FILE
.../project1_windturbine.py

ASSOCIATED FILES
.../WeatherDataFetcher.py

AUTHOR(S)
Mark Janitschek
TH Köln / F09 / IPK / CAISA
MEMS
WS 2023/24

DATE
06.11.2023

LAST MODIFIED
21.11.2023

Copyright 2023 - TH Köln
"""
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
n_M = 0.95  # Wirkungsgrad des Motors
J_G = 2.1 * 10 ** 8  # Massenträgheitsmoment der Gondel
b_G = 2.2 * 10 ** 4  # Reibbeiwert der Gondellagerung
i_G2 = 1000  # Übersetzung (ins Langsame)

# Initialisierung
M_B = [0]  # Bremsmoment
M_G = 0  # Antriebsmoment Gondel

# Input
T = 0.01  # Zeit/Abtastrate
iteration = 300000  # Anzahl Iterationen
v_w = 11  # Windgeschwindigkeit; Möglich eigenen Wert einzutippen, nur aktiv, wenn get_weather = False
w_d = math.radians(0)  # Windrichtung; Möglich eigenen Wert einzutippen, nur aktiv, wenn get_weather = False
get_weather = True  # Wenn True, aktuelle Winddaten werden verwendet

# Output
w = [0]  # Winkelgeschwindigkeit Antriebsstrang
w_ab = [0]  # Winkelgeschwindigkeit Abtriebsstrang
w_G = [0]  # Winkelgeschwindigkeit Gondel

alpha_G_rad = [0]  # Winkel der Gondel in rad
alpha_G_deg = [0]  # Winkel der Gondel in degree
alpha_G_deg_plot = [0]  # Winkel der Gondel in degree immer zwischen 0 und 360 Grad für Plot

iteration_time = [0]  # Zei der Iterationen
delta = [0]  # Winkel zwischen Gondelausrichtung und Windrichtung

P_W = [0]  # Windleistung
P_M = [0]  # Mechanische Leistung des Generators
P_E = [0]  # Elektrische Leistung des Generators
P_G = [0]  # Antriebsleistung Gondel


# ----------------------------------
# FUNCTIONS
# ----------------------------------

# Neues Lambda bestimmen und nur Werte zwischen 5 und 50
def calculate_la(v_w_R, w_prev):
    la_calc = l_R * w_prev / v_w_R  # Berechnet Lambda
    return max(5.0, min(50.0, la_calc))


# c_m berechnen
def calculate_c_m(la_calc):
    c_p_interp = np.interp(la_calc, la, c_p)  # interpoliert den Wert für cp, mithilfe der Leistungsbeiwert-Schnelllaufzahl-Charakteristik und dem berechneten Lambda
    return c_p_interp / la_calc


# Wenn der Winkel zwischen Gondel und Windrichtung >20 oder <-20 ist, wird das Antriebsmoment für die Gondel auf den entsprechenden Wert gesetzt.
def update_M_G(delta_current):
    delta.append(delta_current)
    if delta_current > 20:
        M_G = 1
    elif delta_current < -20:
        M_G = -1
    else:
        M_G = 0
    return M_G


# Print aktuelle Windlage
def print_wind():
    print('--------------AKTUELLE WINDLAGE--------------')
    print('Windrichtung:', math.degrees(w_d), '°')
    print('Windgeschwindigkeit:', v_w, 'm/s')
    print('')


# Print Windradinfos zum Endzeitpunkt
def print_turbine_info():
    print('--------------WINDRAD ZUM ENDZEITPUNKT', iteration_time[iteration], 's--------------')
    print('Windradausrichtung:', alpha_G_deg_plot[iteration], '°')
    print('Wirkende Windgeschwindigkeit:', v_w_R, 'm/s')


# Bremsmoment bestimmen
def break_M_B(w_current):
    if w_current * 60 / (2 * math.pi) > 40:  # Wenn Umdrehungen pro Minute größer als 40
        M_B = 10 ** 7  # Umso geringer das Bremsmoment, desto öfter wird das Bremsmoment angesetzt an der Antriebswelle
    else:
        M_B = 0
    return M_B


# ----------------------------------
# PREPROCESSING
# ----------------------------------

# Aktuelle Winddaten holen
if get_weather:
    weatherdata = WeatherDataFetcher()
    weatherdata.save_weather_data()
    v_w = weatherdata.saved_windspeed / 3.6
    w_d = weatherdata.saved_winddirection  # in [rad]


# ----------------------------------
# MAINPROCESSING
# ----------------------------------

# Main Schleife
if v_w >= 5:  # Windgeschwindigkeit muss mindestens 5 m/s betragen
    for i in range(1, iteration + 1):
        # Bestimmung der wirkenden Windgeschwindigkeit
        delta_current = (math.degrees(w_d) - alpha_G_deg[i - 1] + 180) % 360 - 180  # Bringt die Differenz auf einen Wert zwischen -180 und 180
        v_w_R = v_w * math.cos(math.radians(delta_current))  # Berechnet die auf das Rotorblatt wirkende Windgeschwindigkeit

        # Neues c_m bestimmen
        la_calc = calculate_la(v_w_R, w[i - 1])
        c_m = calculate_c_m(la_calc)

        # Wirkende Windgeschwindigkeit nur über 0 m/s, da sonst negative Windleistung (Wind trifft auf Windrad von hinten)
        if v_w_R < 0:
            v_w_R = 0

        # Bremsmoment, wenn RPM > 40
        M_B.append(break_M_B(w[i - 1]))

        # Berechnet die aktuelle Winkelgeschwindigkeit des Antriebsstrangs [rad/s]
        w.append((T / (J_0 + (J_1 / i_G1 ** 2)) * (c_m * 0.5 * rho_air * math.pi * l_R ** 3 * v_w_R ** 2 - w[i - 1] * (b_0 + (b_1 / i_G1 ** 2) + (K_m / i_G1 ** 2)) - M_B[i])) + w[i - 1])

        # Gondelnachführung
        w_G.append((M_G * 1000 * T) / J_G - (b_G * w_G[i - 1] * T) / J_G + w_G[i - 1])  # Berechnet die Winkelgeschwindigkeit der Gondel [rad/s]
        alpha_G_rad.append(w_G[i] * T + alpha_G_rad[i - 1])  # Berechnet den Winkel der Gondel [rad]
        alpha_G_deg.append(math.degrees(alpha_G_rad[i]))  # Berechnet den Winkel der Gondel [°]
        alpha_G_deg_plot.append(alpha_G_deg[i] % 360)  # Winkel Gondel auf Bereich zwischen 0 und 360 Grad

        # Antriebsmoment für Gondel bestimmen
        if v_w > 25:  # Wenn Windgeschwindigkeit über 25 m/s, Anlage wird aus Wind gedreht
            w[i] = 0  # Anlage wird verriegelt
            if -90.1 < delta_current < -89.9 or 89.9 < delta_current < 90.1:  # Gondel bleibt bei 90° zu Wind stehen, genauer zwischen 89.9° und 90.1°, daher manchmal minimale Windleistung
                w_G[i] = 0
            elif -180 <= delta_current <= -90.1 or 0 <= delta_current <= 89.9:
                M_G = -1
            elif -89.9 <= delta_current <= 0 or 90.1 <= delta_current <= 180:
                M_G = 1
        else:  # Sonst Gondel normal zum Wind hindrehen
            M_G = update_M_G(delta_current)

        # Leistungen berechnen für Generatorstrang
        P_W.append(0.5 * rho_air * math.pi * l_R ** 2 * v_w_R ** 3)  # Berechnet die Windleistung
        w_ab.append(w[i] / i_G1)  # Berechnet die Winkelgeschwindigkeit des Abtriebsstranges [rad/s]
        P_M.append(K_m * (w_ab[i]) * (w_ab[i]))  # Berechnet die mechanische Leistung des Generators
        P_E.append(n_G * P_M[i])  # Berechnet die elektrische Leistung des Generators

        # Leistung für Gondel
        P_G.append(w_G[i] * M_G * n_M)  # Berechne die Antriebsleistung für die Gondel

        # Für Erstellung der Plots
        iteration_time.append(T * i)


# ----------------------------------
# POSTPROCESSING
# ----------------------------------

# Ausgabe wichtiger Werte
if 5 <= v_w <= 25:  # Anlage wird eingeschaltet
    print('--------------ANLAGE--------------')
    print('Anlage wurde eingeschaltet. Windgeschwindigkeit zwischen 5 m/s und 25 m/s.')
    print('')

    print_wind()
    print_turbine_info()

    print('Winkelgeschwindigkeit Antrieb:', w[i], 'rad/s')
    print('Elektrische Leistung des Generators (P\u2091\u2097):', P_E[i], 'W')
elif v_w > 25:  # Anlage wird nicht eingeschaltet über 25 m/s
    print('--------------ANLAGE--------------')
    print('Anlage wurde  nicht eingeschaltet. Windgeschwindigkeit über 25 m/s. Anlage wurde zu 90° aus dem Wind gedreht und verriegelt.')
    print('')

    print_wind()
    print_turbine_info()
elif v_w < 5:  # Anlage wird nicht eingeschaltet unter 5 m/s
    print('--------------ANLAGE--------------')
    print('Anlage wurde nicht eingeschaltet. Windgeschwindigkeit unter 5 m/s. Anlage wurde mechanisch verriegelt.')
    print('')

    print_wind()

# Figure erstellen für Diagramme
fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

# Plot 1: Winkelgeschwindigkeiten
axs[0].plot(iteration_time, w, label='Antriebswelle \u03C9_0')
axs[0].plot(iteration_time, w_G, label='Gondel \u03C9_G')  # Winkelgeschwindigkeit Gondel; sehr gering, in Plot nicht gut erkennbar
axs[0].set_title("Winkelgeschwindigkeiten")
axs[0].set_xlabel("Zeit [s]")
axs[0].set_ylabel("Winkelgeschwindigkeit [rad/s]")
axs[0].legend()

# Plot 2: Leistungen
axs[1].plot(iteration_time, P_W, label='P_W')  # Windleistung
axs[1].plot(iteration_time, P_M, label='P_M')  # Mechanische Leistung Generator
axs[1].plot(iteration_time, P_E, label='P_E')  # Elektrische Leistung Generator
axs[1].plot(iteration_time, P_G, label='P_G')  # Elektrische Leistung Gondelmotor; sehr gering, in Plot nicht gut erkennbar
axs[1].set_title("Leistungen")
axs[1].set_xlabel("Zeit [s]")
axs[1].set_ylabel("Leistung [W]")
axs[1].legend()

# Plot 3: Wind- und Gondelausrichtung
axs[2].plot(iteration_time, [math.degrees(w_d)] * len(iteration_time), label='Windrichtung')
axs[2].plot(iteration_time, alpha_G_deg_plot, label='Gondelrichtung')
axs[2].set_title("Wind- und Gondelausrichtung")
axs[2].set_xlabel("Zeit [s]")
axs[2].set_ylabel("Richtung [°]")
axs[2].set_yticks(np.arange(0, 400, 40))
axs[2].legend()

# Plot 4: Bremsmoment
axs[3].plot(iteration_time, M_B, label='Bremsmoment')
axs[3].set_title("Bremse")
axs[3].set_xlabel("Zeit [s]")
axs[3].set_ylabel("Moment [Nm]")
axs[3].legend()

# Einstellungen für das gesamte Diagramm
plt.tight_layout()
plt.show()
