import control as co
import numpy as np
import matplotlib.pyplot as plt

num = np.array([1])
G_pt1 = np.array([5, 1])
G_pt2a = np.array([5 * 0.05, 5 + 0.05, 1])
G_pt2d = np.array([5 ^ 2, 2 * 0.7 * 5, 1])
G_I = np.array([5, 0])

G_pt1_tf = co.tf(num, G_pt1)
G_pt2a_tf = co.tf(num, G_pt2a)
G_pt2d_tf = co.tf(num, G_pt2d)
G_I_tf = co.tf(num, G_I)


print(G_pt1_tf)
print(G_pt2a_tf)
print(G_pt2d_tf)
print(G_I_tf)

print('Poles von G_pt1:', co.poles(G_pt1_tf))
print('Zeros von G_pt1:', co.zeros(G_pt1_tf))
print('Poles von G_pt2a:', co.poles(G_pt2a_tf))
print('Zeros von G_pt2a:', co.zeros(G_pt2a_tf))
print('Poles von G_pt2d:', co.poles(G_pt2d_tf))
print('Zeros von G_pt2d:', co.zeros(G_pt2d_tf))
print('Poles von G_I:', co.poles(G_I_tf))
print('Zeros von G_I:', co.zeros(G_I_tf))

if all(np.real(co.poles(G_pt2a_tf)) < 0):
    print("Stabil")

co.pzmap(G_pt2a_tf, plot=True)
plt.show()
