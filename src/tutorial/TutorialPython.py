'''
This script is for tutorial purposes.


'''
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from GeneralClass import GeneralClass
import sympy as sy

# Functions
#-----------------------------------------
def GenerateOutput(strOutput=10):
    print(strOutput)

# Sequential programming: simple operations
#------------------------------------------
# Simple data structures
a = 1 + 1
b = 3*2

print(str(a))
print(str(b))

b = a
a = 10

print(str(a))
print(str(b))

f1 = ['a', 3, 8.0]
f2 = [[1, 2],[3, 4],[5, 6]]

e = np.array([1, 3, 4])

print(f1)
print(f2)
print(e)

# Simple function calls
GenerateOutput(15)
GenerateOutput()

print(f1[1])

# Conditional structures
for i in range(len(f1)):
    print(f1[i])

i = 0
while i < 5:
    print(i)
    i += 1

if a > 2:
    print("a is bigger than 2.")
else:
    print("a is NOT bigger than 2.")

if a > 10:
    print("a is bigger than 10.")
elif a > 8:
    print("a is bigger than 8.")
elif a > 6:
    print("a is bigger than 6.")
else:
    print("Whatever")

g1 = list()
h1 = set()
i1 = tuple()
l1 = dict()

g2 = [1,2,3,4]
h2 = {1,2,3,3}
i2 = (1,2)
l2 = {1: 'a', 2: 'b'}

print("This is the value of a: " + str(a))

# OOP
#------------------------------------------
#m = GeneralClass(valc='bla2', valb=17, vala=18)
m = GeneralClass()
m.print_static_values()
m.multiply_with_constant(20)
print(m.val5)

pass