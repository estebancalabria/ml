# Clase Tres - 14 de Julio del 2026

# Repaso

* ML
  * Regresion
    * Algoritmos
      * Regresion lineal
    * Metricas
      * Error
  * Clasificacion
    * Algoritmos
      * Regresion Logistico
      * Basados en Arboles de Decision (Decission Tree)
      * KNN
    * Metricas
      * Matriz de Confusion
* Liberia de ML
  * MatPotLib
    * Seaborn
  * Pandas
  * Sckit-Learn
* Entornos de ML
  * Local VS Code
  * Google Colab
  * Azure ML Studio
  * AWS --> ???
  * Kaggle
* Proceso de ML
* Visualizacion de Datos
  * Histograma
    * Ejemplo de los dados (campana)
  * Plot (Lineas)
  * BoxPlot (Velas)
  * Scatter (puntitos)
  * HeatMap
* Matriz de Confusion

---

# Regresion

* Vamos a hacer un ejemplo de regresion

* En este caso vamos numpy, se puede instalar si no viene (aunque siempre viene instalado)

```python
!pip install numpy
```

* Defino el Dataset
```
import numpy as np

horas_estudio_a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9,10])
notas_a = np.array([1.5, 2.3, 2.5, 3.2, 5.5, 6, 7.3, 8.1, 8.5, 9.5])

horas_estudio_a
```

* Lo muestra

```
array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
```

> [!NOTE]
> En este caso use Numpy para que se manejen los arrays mas eficientemente
> El array asi como esta no funciona en scikit-learn hay que pasarlo de [1,2,3,4] a [ [1], [2], [3], [4] ]

* Para que podamos usar los features en sciit learn los tengo que tranformar con reshape

```
import numpy as np

horas_estudio_a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9,10])
notas_a = np.array([1.5, 2.3, 2.5, 3.2, 5.5, 6, 7.3, 8.1, 8.5, 9.5])

#OJO: 
print(horas_estudio_a)

horas_estudio_a = horas_estudio_a.reshape(-1, 1)

print(horas_estudio_a)
```

* Entrenamos un modelo de regresion Lineal

```
from sklearn.linear_model import LinearRegression

X = horas_estudio_a
y = notas_a

modelo_a = LinearRegression()
modelo_a.fit(X, y)

y_pred = modelo_a.predict(X)

print(y_pred)
print(y)
```

* Hago un gratico con matplotlib y la ayuda de la IA

```python
import matplotlib.pyplot as plt


# Gráfico
plt.figure(figsize=(8,5))

# Puntos reales
plt.scatter(X, y, color="blue", label="Datos reales")

# Línea de regresión
orden = np.argsort(X.flatten())
plt.plot(X.flatten()[orden],
         y_pred[orden],
         color="red",
         linewidth=2,
         label="Regresión lineal")

plt.xlabel("Horas de estudio")
plt.ylabel("Nota")
plt.title("Regresión Lineal: Horas de estudio vs Nota")
plt.legend()
plt.grid(True)

plt.show()
```

## Metricas

* Tres metricas de error

## Colinealidad

## Reduccion de la dimensionalidad

---

# Clasificacion

## Metricas
