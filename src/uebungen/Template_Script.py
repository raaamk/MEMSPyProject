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
.../[File name].py

ASSOCIATED FILES
-

AUTHOR(S)
[Author name]
TH Köln / F09 / IPK / CAISA
MEMS
WS 2023/24

DATE
[Date of editing]

LAST MODIFIED
[Date of change]

Copyright 2023 - TH Köln
'''
#----------------------------------
# DEPENDENCIES
#----------------------------------

# import time
# import math
# import numpy as np
# import pandas as pd
# import control as co
# import scipy as sc
# import sympy as sy
# import matplotlib.pyplot as plt

#----------------------------------
# PARAMETERS
#----------------------------------

# Simulation
#...

# Plant
#...

# Controller
#...

# ML
#...

# Optimization
#...

#...

#----------------------------------
# FUNCTIONS
#----------------------------------

def function_template(parm_1, parms_2, parms_3):
    '''This is a docstring for a short description of the functions purpose. Substitute function name with a meaningful one 
    and keyword 'pass' with code. Use comments.'''
    pass

#----------------------------------
# PREPROCESSING
#----------------------------------

# Execute preprocessing operation
#...



#----------------------------------
# MAINPROCESSING
#----------------------------------

# Execute main operation
#...

#----------------------------------
# POSTPROCESSING
#----------------------------------

# Initialize window for step responses
fig1, ax1 = plt.subplots(nrows=2,ncols=2)
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