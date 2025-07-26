import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

#define the vertices of the triangular domain
vertices = np.array([
    [0, 0],  #vertex A
    [1, 0],  #vertex B
    [0.5, 1]  #vertex C
])

import numpy as np
import matplotlib.pyplot as plt

n = 10  #grid resolution

x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
X, Y = np.meshgrid(x, y)

mask = X + Y <= 1

triangular_points = np.array([(X[i, j], Y[i, j]) for i in range(n) for j in range(n) if mask[i, j]])
triangular_matrix = np.full((n, n, 2), np.nan)  # To preserve grid shape
triangular_matrix[mask] = triangular_points

plt.figure(figsize=(6, 6))
plt.scatter(triangular_points[:, 0], triangular_points[:, 1], color='red')
plt.title("Sampled Points in Triangular Domain")
plt.xlabel("X")
plt.ylabel("Y")
plt.gca().set_aspect('equal')
plt.grid(True)
plt.show()

print("Triangular Matrix of Points:")
print(triangular_matrix)

def static_pressure(P_t, M, gamma):
    return P_t * (1 + (gamma - 1) / 2 * M**2)**(-gamma / (gamma - 1))

P_t = 1585624.603400651  # Stagnation pressure in Pa
M = 2.97       # Mach number
gamma = 1.4   # Specific heat ratio for air

print(static_pressure(P_t, M, gamma))