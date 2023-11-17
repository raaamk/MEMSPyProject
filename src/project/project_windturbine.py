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
rho_air = 1.225
la = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 50.0]
c_p = [0.4, 0.45, 0.48, 0.46, 0.44, 0.4, 0.33, 0.0]
K_m = 75.648

# Geometrie
l_R = 40

# Generatorstrang
n_G = 0.97
J_0 = 6 * 10 ** 7
J_1 = 4 * 10 ** 7
J_Ges = J_0 + J_1 * 100
b_0 = 1.2 * 10 ** 7
b_1 = 10.2 * 10 ** 4
b_Ges = b_0 + b_1 * 100
i_G1 = 1 / 100

# Windnachführung
n_M = 0.95
J_G = 2.1 * 10 ** 8
b_G = 2.2 * 10 ** 4
i_G2 = 1000

# Input
v_w = 25.0  # Windgeschwindigkeit
T = 1  # Abtastzeit
M_B = 0  # Bremsmoment
M_G = 1000  # Windnachführung Motormoment
iteration = 100  # Anzahl Iterationen
w_d = 0  # Windrichtung in Grad

# Output
w = [0.0]
w_G = [0.0]
alpha_G_rad = [0.0]
alpha_G_deg = [0.0]
iteration_time = [0]


# ----------------------------------
# FUNCTIONS
# ----------------------------------


# ----------------------------------
# PREPROCESSING
# ----------------------------------

# Execute preprocessing operation
# c_m = c_p[0]/la[0]
v_w_R = v_w * math.cos(math.radians(w_d))

# ----------------------------------
# MAINPROCESSING
# ----------------------------------

# Execute main operation
for i in range(1, iteration + 1):
    la_calc = l_R * w[i - 1] / v_w

    if la_calc < 5:
        la_calc = 5.0
    elif la_calc > 50:
        la_calc = 50.0

    c_p_interp = np.interp(la_calc, la, c_p)
    c_m = c_p_interp / la_calc

    weatherdata = WeatherDataFetcher()
    weatherdata.save_weather_data()
    v_w = weatherdata.saved_windspeed
    w_d = weatherdata.saved_winddirection
    v_w_R = v_w * math.cos(math.radians(w_d))

    # w.append((c_m * 0.5 * rho_air * math.pi * l_R**3 * v_w_R**2 * T) / (J_0 + J_1 * 100) - w[i-1] * ((K_m * 100 * T) / (J_0 + J_1 * 100) + ((b_0 + b_1 * 100) * T) / (J_0 + J_1 * 100) - 1) - (M_B * T) / (J_0 + J_1 * 100))
    w.append((0.5 * c_m * rho_air * math.pi * l_R ** 3 * v_w_R ** 2 * T) / J_Ges - (w[i - 1] * b_Ges * T) / J_Ges - (K_m * w[i - 1] * 100 * T) / J_Ges + w[i - 1] - (M_B * T) / J_Ges)
    w_G.append((M_G * 1000 * T) / J_G - (b_G * w_G[i - 1] * T) / J_G + w_G[i - 1])
    alpha_G_rad.append(w_G[i] * T + alpha_G_rad[i - 1])
    alpha_G_deg.append(math.degrees(alpha_G_rad[i]))

    if alpha_G_deg[i] - w_d > 20:
        M_G = 10
    elif alpha_G_deg[i] - w_d < 20:
        M_G = -10
    else:
        M_G = 0

    iteration_time.append(T * i)
    print("Durchlauf", i)
    print("w:", w[i])
    print("la_calc:", la_calc)

# print(w)
print("Winkel", math.degrees(alpha_G_rad[iteration]))

# ----------------------------------
# POSTPROCESSING
# ----------------------------------

# Plot w
plt.figure(1)
plt.plot(iteration_time, w)
plt.title("Plot von w")
plt.xlabel("Zeit in Sekunden")
plt.ylabel("Winkelgeschwindigkeit")

# Plot Winkel
plt.figure(2)
plt.plot(iteration_time, alpha_G_deg)
plt.title("Plot von Winkel")
plt.xlabel("Zeit in Sekunden")
plt.ylabel("Winkel")

# Show Plots
plt.show()
