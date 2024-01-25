"""
INFO
Project 3

DESCRIPTION
Abgabe 3 von MEMS Projekt an der TH Köln

FILE
.../project3_ML.py

ASSOCIATED FILES
.../matrix_aufgabe3_daten.csv

AUTHOR(S)
Mark Janitschek
TH Köln / F09 / IPK / CAISA
MEMS
WS 2023/24

DATE
05.01.2024

LAST MODIFIED
25.01.2024

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
import keras as k


# ----------------------------------
# PARAMETERS
# ----------------------------------

t = np.arange(0, 30, 0.01)  # von 0 bis 30 Sekunden in 0.01 Schritten
amp = 1  # Amplitude
freq = 0.1  # Frequenz für Sinus
std_dev = 0.1  # Standardabweichung für Rauschen


# ----------------------------------
# AUFGABE 1
# ----------------------------------

# Übertragungsfunktion definieren
#num = [8.3 * 10 ** (-8)]
num = [8.3 * 10 ** (-0)]  # Ist plausibler ohne 10^-8
den = [5, 1]

# Übertragungsfunktion als System
system_c = co.tf(num, den)
system_c = com.c2d(system_c, 0.01, method='zoh')  # Kontinuierlich zu Zeitdiskret umwandeln mit ZOH


# ----------------------------------
# AUFGABE 2
# ----------------------------------

# Beispiel Eingangssignal erstellen
sinus_signal = amp * np.sin(2 * np.pi * freq * t)  # Sinus-Signal mit Frequenz umd Amplitude
noise = np.random.normal(0, std_dev, sinus_signal.shape)  # Rausch-Signal mit Standardabweichung
#jump_array = [10 ** 8] * 3000  # Sprung-Signal
jump_array = [10 ** 0] * 3000  # Sprung-Signal, angepasst auf ohne 10^-8 in Übertragungsfunktion

# Zusammenfügen der Signale zu einem Eingangssignal
u = (noise + sinus_signal) * jump_array

# System mit Eingangssignal linear simulieren
y, t_out, x_out = com.lsim(system_c, u, t)


# ----------------------------------
# AUFGABE 3
# ----------------------------------

# Variablen erstellen
matrix = []
nu = ny = 11  # n_y = n_u = 11 > 10

# Schleife zum Erstellen der Matrix 24 x 2988
for i in range(1 + ny, 3000):
    new_line = [u[i], u[i - 1], u[i - 2], u[i - 3], u[i - 4], u[i - 5], u[i - 6], u[i - 7], u[i - 8], u[i - 9], u[i - 10], u[i - 11], y[i], y[i - 1], y[i - 2], y[i - 3], y[i - 4], y[i - 5], y[i - 6], y[i - 7], y[i - 8], y[i - 9], y[i - 10], y[i - 11]]
    matrix.append(new_line)

# Matrix wird als csv-Datei gespeichert
np.savetxt("matrix_aufgabe3_daten.csv", matrix, delimiter=',')


# ----------------------------------
# AUFGABE 4
# ----------------------------------

# Matrix aus csv-Datei wieder importieren
matrix_import = np.array(pd.read_csv('matrix_aufgabe3_daten.csv', sep=','))

# Matrix aufteilen in Labels und Features
labels = matrix_import[:, 12]  # 13. Spalte der Matrix sind die Labels y_k
features = np.delete(matrix_import, 12, axis=1)  # Rest sind Features


# ----------------------------------
# AUFGABE 5
# ----------------------------------

# Features und Labels aufteilen in Training + Validierung 90 % (wird zum späteren Zeitpunkt aufgeteilt) und Test 10 %
X_train, X_test, y_train, y_test = skl_ms.train_test_split(features, labels, train_size=0.9, test_size=0.1)


# ----------------------------------
# AUFGABE 6
# ----------------------------------

# Erstellen des sequenziellen Modells mit den einzelnen Layern
model = k.Sequential(
    [
        k.Input(shape=(23, )),  # Input Layer, um den Input-Shape zu übergeben
        k.layers.Normalization(),
        k.layers.Dense(16),
        k.layers.Dense(64, activation='sigmoid'),
        k.layers.Dense(8),
        k.layers.Dense(1)  # Nur ein Wert als Ausgabe
    ]
)


# ----------------------------------
# AUFGABE 7
# ----------------------------------

# Kompilieren des Modells mit Adam-Optimierungsalgorithmus, MSE als Verlustfunktion, MSE als Metrik
model.compile(optimizer='adam', loss='mse', metrics=['mse'])


# ----------------------------------
# AUFGABE 8
# ----------------------------------

# EarlyStopping-Callback erstellen
callback = k.callbacks.EarlyStopping(monitor='loss', patience=3, verbose=1, mode='min', min_delta=0.00001)

# Trainieren des Modells mit Trainingsdatensatz
history = model.fit(
    x=X_train,
    y=y_train,
    batch_size=64,
    epochs=500,
    verbose='auto',
    callbacks=callback,  # Callback nicht verwendet, da sonst zu wenige Epochen; Callback jedoch einsetzbar
    validation_split=0.22222222  # Aufteilen des Trainingsdatensatzes in Training und Validierung (Training 70 %, Validierung 20 %, Test 10 %)
)

# Abspeichern der Verlustwerte
loss_values = history.history['loss']


# ----------------------------------
# AUFGABE 9
# ----------------------------------

# Variablen erstellen
u_predict = [0] * 12 + [1] * 3000  # Eingangsarray erstellen
y_predict = [0] * 12  # Array für die vorhergesagten y-Werte
t_predict = np.arange(0, 10, 0.01)  # von 0 bis 10 Sekunden in 0.01 Schritten, 10 kann angepasst werden damit länger Predict-Schleife läuft

# Schleife zum Vorhersagen von y
print('--------------------------------------------------------------------------------------------------')
for i in range(1 + ny, len(t_predict)):
    print('Predict-Schleife Step:', i - 11)
    input_predict = np.array([[u_predict[i], u_predict[i - 1], u_predict[i - 2], u_predict[i - 3], u_predict[i - 4], u_predict[i - 5], u_predict[i - 6], u_predict[i - 7], u_predict[i - 8], u_predict[i - 9], u_predict[i - 10], u_predict[i - 11], y_predict[i - 1], y_predict[i - 2], y_predict[i - 3], y_predict[i - 4], y_predict[i - 5], y_predict[i - 6], y_predict[i - 7], y_predict[i - 8], y_predict[i - 9], y_predict[i - 10], y_predict[i - 11]]])
    y_predict.append(model.predict(input_predict)[0, 0])  # Da Return von model.predict ndarray, nur Wert wird abgespeichert in y_predict

# Ersten Werte löschen, da 0
y_predict = y_predict[12:]
t_predict = t_predict[12:]

# Evaluation des trainierten Modells mit Testdatensatz
print('--------------------------------------------------------------------------------------------------')
print('Evaluation:')
evaluate_loss, evaluate_metrics = model.evaluate(X_test, y_test, batch_size=64)
print('Verlustwert:', evaluate_loss)
print('Metrikwert:', evaluate_metrics)


# ----------------------------------
# POSTPROCESSING
# ----------------------------------

# Ausgabe einer Zusammenfassung vom Modell
print('--------------------------------------------------------------------------------------------------')
model.summary()

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
axs[1].set_ylabel('Wert')
axs[1].set_title('Ausgang')
axs[1].legend()

# Figure 2 für Aufgabe 8 & 9
fig, axs = plt.subplots(2, 1, figsize=(10, 8))
fig.suptitle('Aufgabe 8 & 9', fontsize=16)

# Plot 1: Verlustwerte
axs[0].plot(loss_values, label='Loss')
axs[0].set_xlabel('Epochen')
axs[0].set_ylabel('Wert')
axs[0].set_title('Trainingsverlauf')
axs[0].legend()

# Plot 2: Vorhergesagten y-Werte vom Modell
axs[1].plot(t_predict, y_predict, label='Predict Sprungantwort')
axs[1].set_xlabel('Zeit [s]')
axs[1].set_ylabel('Wert')
axs[1].set_title('Ausgangssignal')
axs[1].legend()

# Allgemeine Figure-Einstellungen
plt.tight_layout()

# Diagramme zeigen
plt.show()
