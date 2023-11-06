import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
import numpy as np
from scipy.spatial.distance import euclidean

import numpy as np

# Cargar el conjunto de datos Iris (atributos)
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data

# Cargar el conjunto de datos Iris (atributos)
iris = load_iris()
X = iris.data
y = iris.target

colors = ['red', 'purple', 'yellow']
# Crear un gráfico de dispersión 2D con colores personalizados
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=[colors[label] for label in y], edgecolor='k')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("Conjunto de Datos Iris en Espacio de Alta Dimensión")

# Configurar la leyenda para mostrar los nombres de las clases
plt.legend(iris.target_names)

# Mostrar el gráfico
plt.show()

# Crear un gráfico 3D para visualizar los datos en las tres primeras dimensiones
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Asignar colores a las etiquetas automáticamente utilizando un mapa de colores
sc = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis')

ax.set_xlabel(iris.feature_names[0])
ax.set_ylabel(iris.feature_names[1])
ax.set_zlabel(iris.feature_names[2])
ax.set_title("Conjunto de Datos Iris en 3D")

# Agregar barra de color
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Clases')

# Mostrar el gráfico 3D
plt.show()

# Calcular la matriz de similitud inicial en el espacio de alta dimensión
def calculate_initial_similarity_matrix(X):
    n = X.shape[0]
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distance = np.linalg.norm(X[i] - X[j])
            similarity_matrix[i, j] = 1 / (1 + distance)
            similarity_matrix[j, i] = similarity_matrix[i, j]
    return similarity_matrix

# Función para calcular el estrés de Sammon
def sammon_stress(Y, S, n):
    stress = 0
    for i in range(n):
        for j in range(i + 1, n):
            if S[i, j] > 0:
                stress += ((np.linalg.norm(Y[i] - Y[j]) - S[i, j]) ** 2) / S[i, j]
    return stress

np.random.seed(0)
n, m = X.shape
Y = np.random.rand(n, 2)

# Calcular la matriz de similitud inicial en el espacio de alta dimensión
S = calculate_initial_similarity_matrix(X)

# Hiperparámetros del algoritmo
learning_rate = 0.1
max_iterations = 100
tolerance = 1e-5

prev_stress = sammon_stress(Y, S, n)
