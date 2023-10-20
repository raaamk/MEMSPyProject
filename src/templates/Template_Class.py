''' 
INFO
This is a template of a class

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
# CLASS DEFINITION
#----------------------------------
class TemplateClass:

    #----------------------------------
    # VARIABLES
    #----------------------------------
    var_1 = 0
    var_2 = 4.5
    var_3 = 'text'
    var_4 = True
    var_sum = []
	
    #----------------------------------
    # CONSTRUCTOR
    #----------------------------------
    def __init__(self, var_1=1, var_2=5.2, var_3='default',var_4=True):
        self.var_1 = var_1
        self.var_2 = var_2
        self.var_3 = var_3
        self.var_4 = var_4

    #----------------------------------
    # METHODS
    #----------------------------------
    def add_val_to_var2(self, val=0.0):
        var_sum = self.var_2 + val

