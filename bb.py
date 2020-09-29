# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# Make the grid
x, y = np.meshgrid(np.linspace(0, 2, 20),
                      np.linspace(0, 2, 20))
z = 1 / np.sqrt(x**2 + y**2)

# Make the direction data for the arrows
u =    x / (np.sqrt(x**2 + y**2) * np.sqrt((x**2 + y**2)**2 +1))
v =   y / (np.sqrt(x**2 + y**2) * np.sqrt((x**2 + y**2)**2 +1))
w = (x**2 + y**2)/ np.sqrt((x**2 + y**2)**2 +1)

ax.quiver(x, y, z, u, v, w, length=0.4, normalize=True)
ax.plot_wireframe(x, y, z, color='red',linewidth=0.3)

plt.show()