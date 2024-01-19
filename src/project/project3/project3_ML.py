"""
INFO
Project 3

DESCRIPTION
Abgabe 3 von MEMS Projekt an der TH Köln

FILE
.../project3_ML.py

ASSOCIATED FILES
.../

AUTHOR(S)
Mark Janitschek
TH Köln / F09 / IPK / CAISA
MEMS
WS 2023/24

DATE
05.01.2024

LAST MODIFIED
05.01.2024

Copyright 2023 - TH Köln
"""

# ----------------------------------
# DEPENDENCIES
# ----------------------------------

import control as co
import control.matlab as com
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.model_selection as skl_ms
import sklearn as skl

import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import keras as k

# ----------------------------------
# PARAMETERS
# ----------------------------------

t = np.arange(0, 30, 0.01)
amp = 1
freq = 0.1
std_dev = 0.2
jump_sek = 1

# ----------------------------------
# AUFGABE 1
# ----------------------------------

num = [8.3 * 10 ** (-8)]
den = [5, 1]

system_c = co.tf(num, den, 0.01)
# system_d = com.c2d(system_c, 0.01, method='zoh')


# ----------------------------------
# AUFGABE 2
# ----------------------------------

sinus_signal = amp * np.sin(2 * np.pi * freq * t)
noise = np.random.normal(0, std_dev, sinus_signal.shape)
jump_array = [10 ** 8] * 3000

u = (sinus_signal + noise) * jump_array

y, t_out, x_out = com.lsim(system_c, u, t)

# ----------------------------------
# AUFGABE 3
# ----------------------------------

matrix = []
nu = ny = 11

for i in range(1 + ny, 3000):
    new_line = [u[i], u[i - 1], u[i - 2], u[i - 3], u[i - 4], u[i - 5], u[i - 6], u[i - 7], u[i - 8], u[i - 9], u[i - 10], u[i - 11], y[i], y[i - 1], y[i - 2], y[i - 3], y[i - 4], y[i - 5], y[i - 6], y[i - 7], y[i - 8], y[i - 9], y[i - 10], y[i - 11]]
    matrix.append(new_line)

np.savetxt("matrix_aufgabe3_daten.csv", matrix, delimiter=',')

# ----------------------------------
# AUFGABE 4
# ----------------------------------

matrix_import = np.array(pd.read_csv('matrix_aufgabe3_daten.csv', sep=','))
labels = matrix_import[:, 12]
features = np.delete(matrix_import, 12, axis=1)


# ----------------------------------
# AUFGABE 5
# ----------------------------------

X_train, X_test, y_train, y_test = skl_ms.train_test_split(features, labels, train_size=0.9, test_size=0.1)


# ----------------------------------
# AUFGABE 6
# ----------------------------------

model = k.Sequential(
    [
        k.layers.Normalization(input_shape=(23, )),
        k.layers.Dense(16),
        k.layers.Dense(64, activation='sigmoid'),
        k.layers.Dense(8),
        k.layers.Dense(1)
    ]
)


# ----------------------------------
# AUFGABE 7
# ----------------------------------

model.compile(optimizer='adam', loss='mse', metrics=['mse'])


# ----------------------------------
# AUFGABE 8
# ----------------------------------

callback = k.callbacks.EarlyStopping(monitor='loss', patience=3, verbose=1, mode='auto')

history = model.fit(
    x=X_train,
    y=y_train,
    batch_size=64,
    epochs=500,
    verbose='auto',
    callbacks=callback,
    validation_split=0.22222222
)

loss_values = history.history['loss']

print('--------------------------------------------------------------------------------------------------')
model.summary()


# ----------------------------------
# AUFGABE 9
# ----------------------------------

print('--------------------------------------------------------------------------------------------------')
print('Evaluation:')
evaluate_loss, evaluate_metrics = model.evaluate(x=X_test, y=y_test, batch_size=64)
print('Verlustwert:', evaluate_loss)
print('Metrikwert:', evaluate_metrics)

# ----------------------------------
# POSTPROCESSING
# ----------------------------------

# Grid aktivieren für Plots
plt.rcParams['axes.grid'] = True

# Figure 1 für Aufgabe 1 & 2
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig.suptitle('Aufgabe 1 & 2', fontsize=16)

# Plot 1: Eingangssignal
axs[0].plot(t, u, label='Eingangssignal')
axs[0].set_xlabel('Zeit [s]')
axs[0].set_ylabel('Wert')
axs[0].set_title('Eingang')
axs[0].legend()

# Plot 2: Ausgangssignal
axs[1].plot(t_out, y, label='Ausgangssignal')
axs[1].set_xlabel('Zeit [s]')
axs[0].set_ylabel('Wert')
axs[1].set_title('Ausgang')
axs[1].legend()

# Figure 2 für Aufgabe 8 & 9
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig.suptitle('Aufgabe 8 & 9', fontsize=16)

# Plot 1
axs[0].plot(loss_values, label='Loss')
axs[0].set_xlabel('Epochen')
axs[0].set_ylabel('Wert')
axs[0].set_title('Trainingsverlauf')
axs[0].legend()

# Allgemeine Figure-Einstellungen
plt.tight_layout()

# Diagramme zeigen
plt.show()
