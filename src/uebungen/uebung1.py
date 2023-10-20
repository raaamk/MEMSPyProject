import control as co
import numpy as np
import matplotlib.pyplot as plt

s = co.tf('s')

num = np.array([1])
G_pt1 = np.array([5, 1])
G_pt2a = np.array([5 * 0.05, 5 + 0.05, 1])
G_pt2d = np.array([5 ^ 2, 2 * 0.7 * 5, 1])
G_I = np.array([5, 0])


G_pt1_tf = co.tf(num, G_pt1)
G_pt2a_tf = co.tf(num, G_pt2a)
G_I_tf = co.tf(num, G_I)

print(G_pt1_tf)
print(G_pt2a_tf)
print(G_I_tf)

print(co.poles(G_pt1_tf))
print(co.zeros(G_pt1_tf))
print(co.poles(G_pt2a_tf))
print(co.zeros(G_pt2a_tf))

co.pzmap(G_pt2a_tf, plot=True)
plt.show()