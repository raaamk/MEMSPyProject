
# Pakete/Bibliotheken importieren die genutzt werden
import control as co
import numpy as np
import matplotlib.pyplot as plt

#num = Zähler, alle folgenden Nenner
num = np.array([1])
G_pt1 = np.array([5, 1])
G_pt2a = np.array([5 * 0.05, 5 + 0.05, 1])
G_pt2d = np.array([5 ** 2, 2 * 0.7 * 5, 1])
G_I = np.array([5, 0])

#Zusammenfügen von Zähler und den Nennern
G_pt1_tf = co.tf(num, G_pt1)
G_pt2a_tf = co.tf(num, G_pt2a)
G_pt2d_tf = co.tf(num, G_pt2d)
G_I_tf = co.tf(num, G_I)

# Alle Übertragungsfunktionen ausgeben lassen
print(G_pt1_tf)
print(G_pt2a_tf)
print(G_pt2d_tf)
print(G_I_tf)

# Pol- und Nullstellen der Übertragungsfunktionen ausgeben lassen
print('Poles von G_pt1:', co.poles(G_pt1_tf))
print('Zeros von G_pt1:', co.zeros(G_pt1_tf))
print('Poles von G_pt2a:', co.poles(G_pt2a_tf))
print('Zeros von G_pt2a:', co.zeros(G_pt2a_tf))
print('Poles von G_pt2d:', co.poles(G_pt2d_tf))
print('Zeros von G_pt2d:', co.zeros(G_pt2d_tf))
print('Poles von G_I:', co.poles(G_I_tf))
print('Zeros von G_I:', co.zeros(G_I_tf))

# Wenn ALLE realen Polstellen der Übertragungsfunktion G_pt2a negativ sind, "Stabil!" ausgeben lassen
if all(np.real(co.poles(G_pt2a_tf)) < 0):
    print("Stabil!")

# Plotte in:
# figure 1: Polstellen von G_pt2d
# figure 2: Polstellen von G_pt2a
# figure 3: Bode-Diagramm von G_I
# figure 4: Nyquist-Diagramm von G_I
plt.figure(1)
co.pzmap(G_pt2d_tf, plot=True)
plt.figure(2)
co.pzmap(G_pt2a_tf, plot=True)
plt.figure(3)
co.bode(G_I_tf)
plt.figure(4)
co.nyquist_plot(G_I_tf)

# Sprungantwort der einzelnen Uebertragungsfuntkionen erstellen
G_pt1_step = co.step_response(G_pt1_tf)
G_pt2a_step = co.step_response(G_pt2a_tf)
G_pt2d_step = co.step_response(G_pt2d_tf)
G_I_step = co.step_response(G_I_tf)

# In ein 2x2 Diagramm die einzelnen Sprungantworten eintragen
fig5, stepres = plt.subplots(2, 2)
stepres[0, 0].plot(G_pt1_step[0], G_pt1_step[1])
stepres[0, 1].plot(G_pt2a_step[0], G_pt2a_step[1])
stepres[1, 0].plot(G_pt2d_step[0], G_pt2d_step[1])
stepres[1, 1].plot(G_I_step[0], G_I_step[1])

plt.show()