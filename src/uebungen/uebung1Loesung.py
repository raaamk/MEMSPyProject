''' 
INFO
This script containes the first exercise of the MEMS project.

SYNTAX
-

INPUT VARIABLES
-

OUTPUT VARIABLES
-
DESCRIPTION
This script containes the first exercise of the MEMS project.

SEE ALSO
-

FILE
.../Task1.py

ASSOCIATED FILES
-

AUTHOR(S)
Konstantinos Papadopoulos
TH Köln / F09 / IPK / CAISA
MEMS
WS 2023/24

DATE
2023-10-13

LAST MODIFIED
-

Copyright 2023 - TH Köln
'''
#----------------------------------
# DEPENDENCIES
#----------------------------------

import math
import numpy as np
import pandas as pd
import control as co
import scipy as sc
import sympy as sy
import matplotlib.pyplot as plt

#----------------------------------
# PARAMETERS
#----------------------------------

# Simulation definitions
t_start = 0.0
t_stop = 30.0
t_step = 0.01

# Plant definitions
num_pt1 = 1.0
den_pt1 = np.array([5.0, 1.0])
num_pt2_a = 1.0
den_pt2_a = np.array([5.0*0.05, 5.0+0.05, 1.0])
num_pt2_d = 1.0
den_pt2_d = np.array([5.0**2, 2*0.7*5.0, 1.0])
num_i = 1.0
den_i = np.array([5.0, 0.0])
tn_it1 = 0.2
t1_it1 = 5.0

#----------------------------------
# FUNCTIONS
#----------------------------------

def pt1_rec(K, T1, T, x):
    '''Compute outputs of first-order lag (PT1) using recursive formula.'''
    n = len(x)
    y = [0 for i in range(n)]
    for i in range(n):
        if i > 0:
            y[i] = (2*T1/T - 1)/(2*T1/T + 1)*y[i-1] + K/(2*T1/T + 1)*(x[i] + x[i-1])
    return y

def show_eq_symb_rec_electrical_current():
    z, L, R, T, I, e_K, u_K = sy.symbols('z L R T I e_K u_K')
    expd_exp_2 = sy.expand(L*(z - 1)/T*I + R*I - e_K + u_K)
    sol_expd_exp_2 = sy.solve(expd_exp_2, z*I)
    sol_expd_exp_2 = sy.collect(sol_expd_exp_2[0], I)
    print('-------------------------------------------------')
    print('Symbolic recursive equation of electrical current')
    print('-------------------------------------------------')
    print(str(sol_expd_exp_2))

#----------------------------------
# PREPROCESSING
#----------------------------------

# Set time vector
t = np.arange(start=t_start,stop=t_stop, step=t_step)

# Define transfer functions using single function call based on numerator and denominator
sys_pt1 = co.tf(num_pt1, den_pt1)
sys_pt2_a = co.tf(num_pt2_a, den_pt2_a)
sys_pt2_d = co.tf(num_pt2_d, den_pt2_d)
sys_i = co.tf(num_i, den_i)

# Define transfer functions using symbolic variable and algebra
s = co.tf('s')
sys_it1 = 1/(tn_it1*s)*1/(t1_it1*s + 1)

# Define transfer functions using single function call based on zero-pole-gain definition
num, den = sc.signal.zpk2tf([], np.roots(den_pt1), num_pt1)
den = den/den[-1]
sys_pt1_zpk = co.tf(num, den)
num, den = sc.signal.zpk2tf([], np.roots(den_pt2_a), num_pt2_a)
den = den/den[-1]
sys_pt2_a_zpk = co.tf(num, den)
num, den = sc.signal.zpk2tf([], np.roots(den_pt2_d), num_pt2_d)
den = den/den[-1]
sys_pt2_d_zpk = co.tf(num, den)

# Show transfer functions
print('------------------')
print('TRANSFER FUNCTIONS')
print('------------------')
print(sys_pt1)
print(sys_pt2_a)
print(sys_pt2_d)
print(sys_i)
print(sys_it1)

print(sys_pt1_zpk)
print(sys_pt2_a_zpk)
print(sys_pt2_d_zpk)

# Show transfer functions
print('------------------')
print('ZERO_POLE')
print('------------------')

# Get poles for stability checks
pole_pt1 = co.pole(sys_pt1)
pole_pt2_a = co.pole(sys_pt2_a)
pole_pt2_d = co.pole(sys_pt2_d)
pole_i = co.pole(sys_i)
pole_it1 = co.pole(sys_it1)

print('------------------')
print('POLES')
print('------------------')
print("PT1: " + str(pole_pt1))
print("PT2-aperiod.: " + str(pole_pt2_a))
print("PT1-damped: " + str(pole_pt2_d))
print("I: " + str(pole_i))
print("IT1: " + str(pole_it1))

# Get zeros
zero_pt1 = co.zero(sys_pt1)
zero_pt2_a = co.zero(sys_pt2_a)
zero_pt2_d = co.zero(sys_pt2_d)
zero_i = co.zero(sys_i)
zero_it1 = co.zero(sys_it1)

print('------------------')
print('ZEROS')
print('------------------')
print("PT1: " + str(zero_pt1))
print("PT2-aperiod.: " + str(zero_pt2_a))
print("PT1-damped: " + str(zero_pt2_d))
print("I: " + str(zero_i))
print("IT1: " + str(zero_it1))

# Show pole-zero maps
pole, zero = co.pzmap(sys_pt1)

# Get data for bode plots
omega_start = 0.01
omega_stop = 100
step = 0.01
N = int ((omega_stop - omega_start )/step) + 1
omega = np.linspace (omega_start , omega_stop , N)

mag_pt1, pha_pt1, omega_pt1 = co.bode(sys_pt1, omega)
mag_pt2_a, pha_pt2_a, omega_pt2_a = co.bode(sys_pt2_a, omega)
mag_pt2_d, pha_pt2_d, omega_pt2_d = co.bode(sys_pt2_d, omega)
mag_i, pha_i, omega_i = co.bode(sys_i, omega)
mag_it1, pha_it1, omega_it1 = co.bode(sys_it1, omega)

# Convert data of bode plots (magnitude to db and phase to degrees)
mag_pt1_db = 20*np.log10(mag_pt1)
pha_pt1_deg = pha_pt1*180/math.pi

mag_pt2_a_db = 20*np.log10(mag_pt2_a)
pha_pt2_a_deg = pha_pt2_a*180/math.pi

mag_pt2_d_db = 20*np.log10(mag_pt2_d)
pha_pt2_d_deg = pha_pt2_d*180/math.pi

mag_i_db = 20*np.log10(mag_i)
pha_i_deg = pha_i*180/math.pi

mag_it1_db = 20*np.log10(mag_it1)
pha_it1_deg = pha_it1*180/math.pi

# Get data for nyquist plots
#real_pt1, imag_pt1, fre_pt1 = co.nyquist_plot(sys_pt1, omega)
ct_pt1, nq_pt1 = co.nyquist_plot(sys_pt1, plot=False, return_contour=True)
ct_pt2_a, nq_pt2_a = co.nyquist_plot(sys_pt2_a, plot=False, return_contour=True)
ct_pt2_d, nq_pt2_d = co.nyquist_plot(sys_pt2_d, plot=False, return_contour=True)
ct_i, nq_i = co.nyquist_plot(sys_i, plot=False, return_contour=True)
ct_it1, nq_it1 = co.nyquist_plot(sys_it1, plot=False, return_contour=True)

#----------------------------------
# MAINPROCESSING
#----------------------------------

# Determine step responses
t_pt1, y_pt1 = co.step_response(sys=sys_pt1)
t_pt2_a, y_pt2_a = co.step_response(sys=sys_pt2_a)
t_pt2_d, y_pt2_d = co.step_response(sys=sys_pt2_d)
t_i, y_i = co.step_response(sys=sys_i)
t_it1, y_it1 = co.step_response(sys=sys_it1)

# Determine step response using time-discrete, recursive functions
y_pt1_rec = pt1_rec(num_pt1, den_pt1[0], t_step, np.ones(len(t)))

# Get symbolic expression
show_eq_symb_rec_electrical_current()

#----------------------------------
# POSTPROCESSING
#----------------------------------

# Initialize window for step responses
fig1, ax1 = plt.subplots(nrows=2,ncols=2)
fig1.canvas.manager.set_window_title('Step responses')

# Show step responses
ax1[0,0].plot(t_pt1, y_pt1, label="PT1")
ax1[0,0].plot(t, y_pt1_rec, label="PT1 (rec.)")
ax1[0,0].set_title("Step response")
ax1[0,0].set_xlabel("Time (t) / s")
ax1[0,0].set_ylabel("Output (y) / s")
ax1[0,0].legend()
ax1[0,0].grid()

ax1[0,1].plot(t_pt2_a, y_pt2_a, label="PT2-aperiod.")
ax1[0,1].plot(t_pt2_d, y_pt2_d, label="PT1-damped")
ax1[0,1].set_title("Step response")
ax1[0,1].set_xlabel("Time (t) / s")
ax1[0,1].set_ylabel("Output (y) / s")
ax1[0,1].legend()
ax1[0,1].grid()

ax1[1,0].plot(t_i, y_i, label="I")
ax1[1,0].plot(t_it1, y_it1, label="IT1")
ax1[1,0].set_title("Step response")
ax1[1,0].set_xlabel("Time (t) / s")
ax1[1,0].set_ylabel("Output (y) / s")
ax1[1,0].legend()
ax1[1,0].grid()

fig1.show()

# Initialize window for pole-zero maps
fig2, ax2 = plt.subplots(nrows=2,ncols=2)
fig2.canvas.manager.set_window_title('Pole-zero maps')

# Show pole-zero map
ax2[0,0].plot(0, 0, marker='+', color='k',)
ax2[0,0].plot(np.real(pole_pt1), np.imag(pole_pt1), marker='+', color='b', label="pole(PT1)")
ax2[0,0].plot(np.real(zero_pt1), np.imag(zero_pt1), marker='o', color='b', label="zero(PT1)")
ax2[0,0].set_title("Pole-zero map")
ax2[0,0].set_xlabel("Real")
ax2[0,0].set_ylabel("Imag.")
ax2[0,0].legend()
ax2[0,0].grid()

ax2[0,1].plot(0, 0, marker='+', color='k',)
for i in range(len(pole_pt2_a)):
    ax2[0,1].plot(np.real(pole_pt2_a[i]), np.imag(pole_pt2_a[i]), marker='+', color='b', label="pole(PT2-aperiod.)")
ax2[0,1].plot(np.real(zero_pt2_a), np.imag(zero_pt2_a), marker='o', color='b', label="zero(PT2-aperiod.)")
for i in range(len(pole_pt2_d)):
    ax2[0,1].plot(np.real(pole_pt2_d[i]), np.imag(pole_pt2_d[i]), marker='+', color='r', label="pole(PT2-damped)")
ax2[0,1].plot(np.real(zero_pt2_d), np.imag(zero_pt2_d), marker='o', color='r', label="zero(PT2-damped)")
ax2[0,1].set_title("Pole-zero map")
ax2[0,1].set_xlabel("Real")
ax2[0,1].set_ylabel("Imag.")
ax2[0,1].legend()
ax2[0,1].grid()

ax2[1,0].plot(0, 0, marker='+', color='k',)
for i in range(len(pole_i)):
    ax2[1,0].plot(np.real(pole_i[i]), np.imag(pole_i[i]), marker='+', color='b', label="pole(I)")
ax2[1,0].plot(np.real(zero_i), np.imag(zero_i), marker='o', color='b', label="zero(I)")
for i in range(len(pole_it1)):
    ax2[1,0].plot(np.real(pole_it1[0]), np.imag(pole_it1[0]), marker='+', color='r', label="pole(IT1)")
ax2[1,0].plot(np.real(zero_it1), np.imag(zero_it1), marker='o', color='r', label="zero(IT1)")
ax2[1,0].set_title("Pole-zero map")
ax2[1,0].set_xlabel("Real")
ax2[1,0].set_ylabel("Imag.")
ax2[1,0].legend()
ax2[1,0].grid()

fig2.show()

# Initialize window for bode plots
fig3, ax3 = plt.subplots(nrows=4,ncols=2)
fig3.canvas.manager.set_window_title('Bode plots')

# Show bode plots
ax3[0,0].semilogx(omega_pt1, mag_pt1_db, label="PT1")
ax3[0,0].set_title("Magnitude")
ax3[0,0].set_xlabel("Time (t) / s")
ax3[0,0].set_ylabel("Magnitude (A) / dB")
ax3[0,0].legend()
ax3[0,0].grid()

ax3[1,0].semilogx(omega_pt1, pha_pt1_deg, label="PT1")
ax3[1,0].set_title("Phase")
ax3[1,0].set_xlabel("Time (t) / s")
ax3[1,0].set_ylabel("Phase (\phi) / °")
ax3[1,0].legend()
ax3[1,0].grid()

ax3[0,1].semilogx(omega_pt2_a, mag_pt2_a_db, label="PT2-aperiod.")
ax3[0,1].semilogx(omega_pt2_d, mag_pt2_d_db, label="PT2-damped")
ax3[0,1].set_title("Magnitude")
ax3[0,1].set_xlabel("Time (t) / s")
ax3[0,1].set_ylabel("Magnitude (A) / dB")
ax3[0,1].legend()
ax3[0,1].grid()

ax3[1,1].semilogx(omega_pt2_a, pha_pt2_a_deg, label="PT2-aperiod.")
ax3[1,1].semilogx(omega_pt2_d, pha_pt2_d_deg, label="PT2-damped")
ax3[1,1].set_title("Phase")
ax3[1,1].set_xlabel("Time (t) / s")
ax3[1,1].set_ylabel("Phase (\phi) / °")
ax3[1,1].legend()
ax3[1,1].grid()

ax3[2,0].semilogx(omega_i, mag_i_db, label="I")
ax3[2,0].semilogx(omega_it1, mag_it1_db, label="IT1")
ax3[2,0].set_title("Magnitude")
ax3[2,0].set_xlabel("Time (t) / s")
ax3[2,0].set_ylabel("Magnitude (A) / dB")
ax3[2,0].legend()
ax3[2,0].grid()

ax3[3,0].semilogx(omega_i, pha_i_deg, label="I")
ax3[3,0].semilogx(omega_it1, pha_it1_deg, label="IT1")
ax3[3,0].set_title("Phase")
ax3[3,0].set_xlabel("Time (t) / s")
ax3[3,0].set_ylabel("Phase (\phi) / °")
ax3[3,0].legend()
ax3[3,0].grid()

fig3.show()

# Initialize window for nyquist plots
fig4, ax4 = plt.subplots(nrows=2,ncols=2)
fig4.canvas.manager.set_window_title('Nyquist plots')

# Show Nyquist plots
ax4[0,0].plot(0, 0, marker='+', color='k')
ax4[0,0].plot(-1, 0, marker='o', color='k')
ax4[0,0].plot(mag_pt1*np.cos(pha_pt1), mag_pt1*np.sin(pha_pt1), label="pole(PT1)")
ax4[0,0].set_title("Nyquist")
ax4[0,0].set_xlabel("Real")
ax4[0,0].set_ylabel("Imag.")
ax4[0,0].legend()
ax4[0,0].grid()


ax4[0,1].plot(0, 0, marker='+', color='k')
ax4[0,1].plot(-1, 0, marker='o', color='k')
ax4[0,1].plot(mag_pt2_a*np.cos(pha_pt2_a), mag_pt2_a*np.sin(pha_pt2_a), label="pole(PT2-aperiod.)")
ax4[0,1].plot(mag_pt2_d*np.cos(pha_pt2_d), mag_pt2_d*np.sin(pha_pt2_d), label="pole(PT2-damped)")
ax4[0,1].set_title("Nyquist")
ax4[0,1].set_xlabel("Real")
ax4[0,1].set_ylabel("Imag.")
ax4[0,1].legend()
ax4[0,1].grid()


ax4[1,0].plot(0, 0, marker='+', color='k')
ax4[1,0].plot(-1, 0, marker='o', color='k')
ax4[1,0].plot(mag_i*np.cos(pha_i), mag_i*np.sin(pha_i), label="pole(I)")
ax4[1,0].plot(mag_it1*np.cos(pha_it1), mag_it1*np.sin(pha_it1), label="pole(IT1)")
ax4[1,0].set_title("Nyquist")
ax4[1,0].set_xlabel("Real")
ax4[1,0].set_ylabel("Imag.")
ax4[1,0].legend()
ax4[1,0].grid()

fig4.show()

plt.waitforbuttonpress(0)