# type: ignore


import numpy as np
import matplotlib.pyplot as plt

U = np.array([3, 4, 5])
V = np.array([2, 3, 2])
W = np.add(U, V)

ax = plt.figure().add_subplot(projection='3d')
ax.set_xlim([0, 10])
ax.set_ylim([0, 10])
ax.set_zlim([0, 10])

 # Plotta i vettori U, V, W dall'origine
ax.quiver([0], [0], [0], [U[0]], [U[1]], [U[2]], label='U', color='r')  # U
ax.quiver([0], [0], [0], [V[0]], [V[1]], [V[2]], label='V', color='g')  # V
ax.quiver([0], [0], [0], [W[0]], [W[1]], [W[2]], label='W', color='b')  # W

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()