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
.../project1.py

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

# import time
import math
# import numpy as np
# import pandas as pd
# import control as co
# import scipy as sc
# import sympy as sy
#import matplotlib.pyplot as plt
import numpy

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
J_0 = 6*10**7
J_1 = 4*10**7
J_Ges = J_0 + J_1 * 100
b_0 = 1.2*10**7
b_1 = 10.2*10**4
b_Ges = b_0 + b_1 * 100
i_G1 = 1/100

# Windnachführung
n_M = 0.95
J_G = 2.1 * 10 ** 8
b_G = 2.2 * 10 ** 4
i_G2 = 1000

# Input
v_w = 25
T = 1
M_B = 0

# Output
w = [0]

# ----------------------------------
# FUNCTIONS
# ----------------------------------

def function_template(parm_1, parms_2, parms_3):
    '''This is a docstring for a short description of the functions purpose. Substitute function name with a meaningful one
    and keyword 'pass' with code. Use comments.'''
    pass


# ----------------------------------
# PREPROCESSING
# ----------------------------------

# Execute preprocessing operation
c_m = c_p[0]/la[0]


# ----------------------------------
# MAINPROCESSING
# ----------------------------------

# Execute main operation
for i in range(1, 1800):
    print(i)
    #w.append((c_m * 0.5 * rho_air * math.pi * l_R**3 * v_w**2 * T) / (J_0 + J_1 * 100) - w[i-1] * ((K_m * 100 * T) / (J_0 + J_1 * 100) + ((b_0 + b_1 * 100) * T) / (J_0 + J_1 * 100) - 1) - (M_B * T) / (J_0 + J_1 * 100))
    w.append((0.5 * c_m * rho_air * math.pi * l_R**3 * v_w**2 * T) / J_Ges - (w[i - 1] * b_Ges * T) / J_Ges - (K_m * w[i - 1] * 100 * T) / J_Ges + w[i - 1] - (M_B * T) / J_Ges)
    print("W:", w[i])
    la_calc = l_R * w[i] / v_w

    if la_calc < 5:
        la_calc = 5
    elif la_calc > 50:
        la_calc = 50

    c_p_interp = numpy.interp(la_calc, la, c_p)
    c_m = c_p_interp / la_calc
    print("la_calc", la_calc)
print(w)
# ----------------------------------
# POSTPROCESSING
# ----------------------------------

# Initialize window for step responses

