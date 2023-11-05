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
