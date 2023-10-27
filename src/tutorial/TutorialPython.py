'''
This script is for tutorial purposes.


'''
import math
import numpy as np
import pandas as pd
import copy
import control as cp
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

# NUMPY
#------------------------------------------

arr_1 = np.array(42)
arr_2 = np.array([1, 2, 3])
b = arr_2[1]

arr_1_2 = arr_1
print(id(arr_1_2))
print(id(arr_1))

arr_1_2 = copy.deepcopy(arr_1)
print(id(arr_1_2))
print(id(arr_1))

arr_2 = np.array([[1, 2],[3, 4]])
print(arr_2)

print(arr_2.shape)
print(arr_2.ndim)

arr_3 = np.array([1, 2, 3])
print(arr_3[-1])
print(arr_3[-2])

arr_4 = np.array([1, 2, 3, 4, 5, 6])
print(arr_4[2:-1])
print(arr_4[2:])
print(arr_4[-3:])
print(arr_4[::2])

arr_5 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(arr_5[:,0])
print(arr_5[-1,:])

arr_6 = np.array([1, 2, 3], dtype="S")
arr_7 = np.array([1, 2, 3], dtype="i4")

arr_8 = np.array([1.1, 2.0, 3.7, 4.2])
arr_9 = np.array([1.1, 2.0, 3.7, 4.2]).astype(int)
arr_9 = np.array([1, 0, 3])

arr_10 = arr_8.copy()
print(id(arr_8))
print(id(arr_10))

arr_11 = arr_8.view()
print(id(arr_8))
print(id(arr_11))

arr_12 = np.array([11, 12, 13, 14, 15, 16, 17, 18])
arr_12_rsh = arr_12.reshape(2,4)
print(arr_12_rsh)
arr_12_rsh_2 = arr_12.reshape(2,2,2)
print(arr_12_rsh_2)

arr_12 = np.array([11, 12, 13, 14, 15, 16, 17, 18])
for x in arr_12:
    print(x)

for x in arr_5:
    for y in x:
        print(y)

arr_13 = np.array([1, 2, 3])
arr_14 = np.array([4, 5, 6])
arr_15 = np.concatenate((arr_13, arr_14), axis=0)
#arr_16 = np.concatenate((arr_13, arr_14), axis=1)
print(arr_15)
#print(arr_16)

arr_17 = np.stack((arr_13, arr_14), axis=0)
arr_18 = np.stack((arr_13, arr_14), axis=1)
print(arr_17)
print(arr_18)

arr_19 = np.hstack((arr_13, arr_14))
arr_20 = np.vstack((arr_13, arr_14))
print(arr_19)
print(arr_20)

arr_21 = np.array_split(arr_12, 3)
print(arr_21)

x = np.where(arr_12 == 15)
print(x)

x_1 = np.where(arr_12%2 == 0)
print(x_1)

arr_21 = np.array([1, 3, 7, 2])
arr_21_sorted = np.sort(arr_21, )

idx_12_gt15 = arr_12 > 15
arr_12_sorted = arr_12[idx_12_gt15]

A_0_1 = np.array([[3, 4], [5, 6]])
A_0_2 = np.array([[8, 9], [10, 11]])
m_A_0_12 = np.multiply(A_0_1, A_0_2)
m_A_0_13 = np.matmul(A_0_1, A_0_2)
m_A_0_14 = A_0_1 @ A_0_2
print(m_A_0_12)
print(m_A_0_13)
print(m_A_0_14)

A_1 = np.array([[1, 2],[3, 4]])
b_1 = np.array([1, 2])
x_1 = np.linalg.solve(A_1, b_1)

mu = 0
sigma = 1 
n = 1000

dist_normal = np.random.normal(loc=mu, scale=sigma, size=n)

# PANDAS
#------------------------------------------

series_object_1 = pd.Series([1, 3, 5, np.nan, 6, 8])
print(series_object_1)

dates_object_1 = pd.date_range("2023-02-08", periods=6)
print(dates_object_1)

data_frame_1 = pd.DataFrame(np.random.randn(6,4), index=dates_object_1, columns=list("ABCD"))
print(data_frame_1)

dict_data_frame_2 = {"A": 1.0, 
                     "B": pd.Timestamp("2023-02-01"),
                     "C": pd.Series(1, index=list(range(4)), dtype="float32"),
                     "D": np.array([3] * 4, dtype="int32"),
                     "E": pd.Categorical(["test", "train", "test", "train"]),
                     "F": "foo"}
data_frame_2 = pd.DataFrame(dict_data_frame_2)
print(data_frame_2)

print(data_frame_2.dtypes)

print(data_frame_2.head(1))
print(data_frame_2.tail(1))
print(data_frame_2.index)
print(data_frame_2.columns)

print(data_frame_2.to_numpy())
print(data_frame_2.describe())

print(data_frame_2.T)

print(data_frame_2.sort_index(axis=1, ascending=False))
print(data_frame_2.sort_values(by="B", ascending=False))

print(data_frame_2["A"])
print(data_frame_2[0:3])

print(data_frame_1.loc[dates_object_1])
print(data_frame_1.loc[:,["A", "B"]])

print(data_frame_1.at["2023-02-08", "A"])

print(data_frame_1.iloc[0, 0])
print(data_frame_1.iloc[3])

print(data_frame_1.iloc[3:5,:])
print(data_frame_1.iloc[:,0:2])

print(data_frame_1[data_frame_1["A"] > 0])
print(data_frame_1[data_frame_1 > 0])

data_frame_1_copy = data_frame_1.copy()
print(id(data_frame_1))
print(id(data_frame_1_copy))

series_object_2 = pd.Series(np.float32([1,2,3,4,5,6]), 
                            index=pd.date_range("2023-02-08", periods=6))
print(series_object_2)
data_frame_1["F"] = series_object_2
print(data_frame_1)

data_frame_1.at[1,1] = 0
data_frame_1.loc[:, "D"] = np.array([5]*len(data_frame_1))
print(data_frame_1)

data_frame_3 = data_frame_1.copy()
data_frame_3[data_frame_3 > 0] = -data_frame_3
print(data_frame_3)

data_frame_4 = data_frame_1.reindex(index=dates_object_1[0:4],
                                    columns=list(data_frame_1.columns) + ["E"])
data_frame_4.loc[dates_object_1[0]:dates_object_1[1], "E"] = 1

print(data_frame_4.dropna(how="any"))

print(data_frame_4.fillna(value=5))
print(data_frame_4.isna())

print(data_frame_1.mean())
print(data_frame_1.max(axis=1))

print(data_frame_1.apply(lambda x: x.max()-x.min()))
series_object_4 = pd.Series(np.random.randint(0, 7, size=10))
print(series_object_4)
print(series_object_4.value_counts)

series_object_5 = pd.Series(["A", "B", "C", "Aaba", np.nan, "CABA", "dog", "cat"])
print(series_object_5.str.lower())

data_frame_5 = pd.DataFrame(np.random.rand(10, 4))
data_frame_5_pieces = [data_frame_5[:3], data_frame_5[3:7], data_frame_5[7:]]
print(data_frame_5_pieces)
data_frame_5_recombined = pd.concat(data_frame_5_pieces)
print(data_frame_5_recombined)

left_data_frame = pd.DataFrame({"key":["foo","bar"], "lval":[1,2]})
right_data_frame = pd.DataFrame({"key":["foo","bar"], "lval":[4,5]})
data_frame_merged = pd.merge(left_data_frame, right_data_frame, on="key")
print(data_frame_merged)

tuples_1 = list(zip(["bar", "bar", "baz", "baz"], ["one", "two", "one", "two"])) 
print(tuples_1)

ts_5 = pd.Series(data=np.random.randn(1000),
                index=pd.date_range("2023-1-1", periods=1000))
ts_5 = ts_5.cumsum()
print(ts_5)
#ts_5.plot()
#plt.show()

data_frame_5.to_csv("foo.csv")
data_frame_6 = pd.read_csv("foo.csv")
print(data_frame_6)

# SYMPY
#------------------------------------------

a, b, c = sy.symbols('a b c')
x1 = (-b + sy.sqrt(b**2 - 4*a*c))/(2*a)
x2 = (-b - sy.sqrt(b**2 - 4*a*c))/(2*a)
print('x1: ' + str(x1))
print('x2: ' + str(x2))

x, y = sy.symbols('x y')
z = (x + 2)*(y + 5)
z_exp = sy.expand(z)
z_fac = sy.factor(z_exp)
print(z_exp)
print(z_fac)

z_exp_subs = z_exp.subs([(x, 2)])
print(z_exp_subs)

z_exp_eval = z_exp.evalf(subs={y: 3.0, x: 6.0})
print(z_exp_eval)

int_z_exp = sy.integrate(z_exp, x)
print(int_z_exp)

diff_z_exp = sy.diff(z_exp, x)
print(diff_z_exp)

eq_exp_2 = sy.Eq(0, 1 + (1+1/x)**x)
print(eq_exp_2)

lmt_exp_2 = sy.limit(eq_exp_2.rhs, x, sy.oo)
print(lmt_exp_2)

ser_exp_2 = sy.series(sy.sin(x), x, 0, 4)
print(ser_exp_2)

rts_exp_2 = sy.solve(x**2 + 5*x - 2, x)
print(rts_exp_2)

f = sy.symbols('f', cls=sy.Function)
diff_eq = sy.Eq(f(x).diff(x,x) - 2*f(x).diff(x) + f(x), sy.cos(x))
sol_exp_3 = sy.dsolve(diff_eq, f(x))
print(sol_exp_3)

smp_exp_1 = sy.simplify(2*x + 5*y + x + x*3 + x**2 + y**2 - 2*y)
print(smp_exp_1)

col_exp_1 = sy.collect(2*x + 5*y + x + x*3 + x**2 + y**2 - 2*y, x, evaluate=False)
print(col_exp_1)

# MATPLOTLIB
#------------------------------------------

x = [0, 1, 2, 3, 4, 5]
y1 = [19.3, 73.2, 46.2, 171.2, 165.6, 820.2]
y2 = [17.3, 73.2, 42.2, 173.2, 163.6, 890.2]

# plt.plot(x,y1, label="f1")
# plt.plot(x,y2, label="f2")
# plt.title("Linieplot")
# plt.xlabel('X-Achse')
# plt.ylabel('Y-Achse')
# plt.show()

fig, ax = plt.subplots(nrows=2, ncols=2)
ax[0,0].plot(x, y1, label="f1")
ax[0,1].plot(x, y2, label="f2")

ax[0,0].set_title("Linieplot 1")
ax[0,0].set_xlabel("X-Achse")
ax[0,0].set_ylabel("Y-Achse")

ax[0,1].set_title("Linieplot 2")
ax[0,1].set_xlabel("X-Achse")
ax[0,1].set_ylabel("Y-Achse")
plt.show()

pass