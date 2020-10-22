# from sympy import *
# from sympy.abc import *
# from sympy.plotting import plot3d

import sympy
import matplotlib.pyplot as plt
import numpy as np
import math

x = sympy.Symbol('x')
y = sympy.Symbol('y')
Q = sympy.Symbol('Q')
l = sympy.Symbol('l')

exq = 3 / sympy.sqrt((x + 1 )**2 + y**2) - 1/sympy.sqrt((x - 1)**2 + y**2)

u = - sympy.diff(exq,x)
v = - sympy.diff(exq,y)

print(str(u) + '\n' + str(v))

from sympy.utilities.autowrap import ufuncify

X, Y = np.meshgrid(np.linspace(-10,10,40), np.linspace(-10,10,40))
uxy = ufuncify((x, y), u)
vxy = ufuncify((x, y), v)
U = uxy(X, Y)/ ((uxy(X, Y)**2 + vxy(X, Y)**2)**(1/2))
V = vxy(X, Y) / ((uxy(X, Y)**2 + vxy(X, Y)**2)**(1/2))
plt.figure()
plt.quiver(X, Y, U, V, scale=25)
plt.show()

# plot(sin(x), (x, -2*pi, 2*pi))

# print(type(d_x))

# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# import numpy as np

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# x, y, z = np.meshgrid(np.linspace(0, 2, 20),
#                       np.linspace(0, 2, 20),
#                       np.linspace(0, 2, 20))

# ax.quiver(x, y, z, d_x, d_y, d_z, length=0.4, normalize=True)
# plt.show()
