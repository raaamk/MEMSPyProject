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
# import math
# import numpy as np
# import pandas as pd
# import control as co
# import scipy as sc
# import sympy as sy
import matplotlib.pyplot as plt

# ----------------------------------
# PARAMETERS
# ----------------------------------

# Umgebung
rho_air = 1.225
la = [5.0, 6.0, 7.0, 8.0, 9.0, 11.0, 50.0]
c_p = [0.4, 0.45, 0.48, 0.46, 0.44, 0.4, 0.33, 0.0]
K_m = 75.648

# Geometrie
l_R = 40

# Generatorstrang
n_G = 0.97
J_0 = 6*10**7
J_1 = 4*10**7
b_0 = 1.2*10**7
b_1 = 10.2*10**4
i_G1 = 1/100

# Windnachführung
n_M = 0.95
J_G = 2.1*10**8
b_G = 2.2*10**4
i_G2 = 1000

# Input
v_w = 1
T = 5
M_B = 1*10**7

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
# ...


# ----------------------------------
# MAINPROCESSING
# ----------------------------------

# Execute main operation


# ----------------------------------
# POSTPROCESSING
# ----------------------------------

# Initialize window for step responses
fig1, ax1 = plt.subplots(nrows=2, ncols=2)
fig1.canvas.manager.set_window_title('Step responses')

# Show results
# ax1[0,0].plot(t_pt1, y_pt1, label="PT1")
# ...
# ax1[0,0].set_title("Title")
# ax1[0,0].set_xlabel("x")
# ax1[0,0].set_ylabel("y")
# ax1[0,0].legend()
# ax1[0,0].grid()

# fig1.show()
plt.waitforbuttonpress(0)
