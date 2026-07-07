# LAB-0800 – Clasificacion - ML: Iris con KNN

## Objetivo

Entrenar un modelo KNN (K-Nearest Neighbors) sobre el dataset Iris, y usar los gráficos característicos de este algoritmo: scatter por clase, frontera de decisión, matriz de confusión y precisión según el valor de k.

## Requisitos previos

```
pip install scikit-learn matplotlib pandas
```

## El dataset

Iris viene incluido en scikit-learn, no hace falta descargar nada. Tiene 150 flores, 4 features (largo y ancho de sépalo, largo y ancho de pétalo) y 3 clases (especies): Setosa, Versicolor, Virginica.

```python
from sklearn.datasets import load_iris

datos = load_iris()
X = datos.data
y = datos.target

print(X[:5])
print(y[:5])
print(datos.target_names)
```

### Salida

```
[[5.1 3.5 1.4 0.2]
 [4.9 3.  1.4 0.2]
 [4.7 3.2 1.3 0.2]
 [4.6 3.1 1.5 0.2]
 [5.  3.6 1.4 0.2]]
[0 0 0 0 0]
['setosa' 'versicolor' 'virginica']
```

- `X` tiene las 4 columnas numéricas (features).
- `y` tiene el número de clase (0 = setosa, 1 = versicolor, 2 = virginica).

---

## Ejercicio 1 – Scatter de dos features, coloreado por clase

Para poder graficar en 2D, usamos solo 2 de las 4 features: largo y ancho de pétalo (columnas 2 y 3).

### Código

```python
import matplotlib.pyplot as plt

largo_petalo = X[:, 2]
ancho_petalo = X[:, 3]

plt.scatter(largo_petalo, ancho_petalo, c=y)
plt.xlabel("Largo de pétalo")
plt.ylabel("Ancho de pétalo")
plt.title("Iris - Largo vs Ancho de pétalo")
plt.show()
```

### Explicación línea por línea

- `largo_petalo = X[:, 2]`: nos quedamos con la columna 2 de `X` (todas las filas, columna en la posición 2), que es el largo de pétalo.
- `ancho_petalo = X[:, 3]`: lo mismo con la columna 3, el ancho de pétalo.
- `plt.scatter(largo_petalo, ancho_petalo, c=y)`: dibuja un punto por flor, usando `largo_petalo` para el eje X y `ancho_petalo` para el eje Y. `c=y` colorea cada punto según su clase.

### Consigna

Mirando el gráfico, ¿alguna de las 3 especies queda claramente separada del resto?

---

## Ejercicio 2 – Entrenar el modelo KNN

Para entrenar el modelo necesitamos las dos columnas juntas en una sola tabla (no separadas como en el ejercicio anterior).

### Código

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X_2d = np.column_stack((largo_petalo, ancho_petalo))

X_train, X_test, y_train, y_test = train_test_split(X_2d, y, test_size=0.3, random_state=42)

modelo = KNeighborsClassifier(n_neighbors=5)
modelo.fit(X_train, y_train)

print(modelo.score(X_test, y_test))
```

### Explicación línea por línea

- `X_2d = np.column_stack((largo_petalo, ancho_petalo))`: junta las dos listas (`largo_petalo` y `ancho_petalo`) en una sola tabla de 2 columnas. El modelo necesita recibir todas las features juntas en una sola estructura, no por separado.
- `train_test_split(X_2d, y, test_size=0.3, random_state=42)`: separa los datos en un conjunto de entrenamiento (70%) y uno de prueba (30%). `random_state=42` hace que la separación sea siempre la misma cada vez que se corre el código.
- `KNeighborsClassifier(n_neighbors=5)`: crea el modelo KNN, indicando que va a mirar los 5 vecinos más cercanos para decidir la clase de un punto nuevo.
- `modelo.fit(X_train, y_train)`: entrena el modelo con los datos de entrenamiento.
- `modelo.score(X_test, y_test)`: calcula el porcentaje de aciertos del modelo sobre los datos de prueba (que no vio durante el entrenamiento).

### Consigna

Cambiá `n_neighbors` a 1 y volvé a correr el entrenamiento. ¿El score sube o baja?

---

## Ejercicio 3 – Frontera de decisión

### Código

```python
import numpy as np

# Creamos una grilla de puntos que cubre todo el gráfico
minimo_x, maximo_x = largo_petalo.min() - 1, largo_petalo.max() + 1
minimo_y, maximo_y = ancho_petalo.min() - 1, ancho_petalo.max() + 1
xx, yy = np.meshgrid(np.arange(minimo_x, maximo_x, 0.1), np.arange(minimo_y, maximo_y, 0.1))

# Le preguntamos al modelo qué clase predice en cada punto de la grilla
puntos_grilla = np.column_stack((xx.ravel(), yy.ravel()))
predicciones_grilla = modelo.predict(puntos_grilla)
predicciones_grilla = predicciones_grilla.reshape(xx.shape)

plt.contourf(xx, yy, predicciones_grilla, alpha=0.3)
plt.scatter(largo_petalo, ancho_petalo, c=y, edgecolor="k")
plt.xlabel("Largo de pétalo")
plt.ylabel("Ancho de pétalo")
plt.title("Frontera de decisión - KNN")
plt.show()
```

### Explicación línea por línea

- `minimo_x, maximo_x = largo_petalo.min() - 1, largo_petalo.max() + 1`: calcula el rango de valores a cubrir en el eje X, con un margen extra de 1 para que los puntos no queden pegados al borde del gráfico.
- `minimo_y, maximo_y = ...`: lo mismo para el eje Y, con `ancho_petalo`.
- `np.meshgrid(...)`: genera una grilla de puntos que cubre todo ese rango, para poder pintar el fondo según la predicción del modelo en cada zona.
- `puntos_grilla = np.column_stack((xx.ravel(), yy.ravel()))`: junta las coordenadas de la grilla en una tabla de 2 columnas, con el mismo formato que espera el modelo (igual que hicimos con `X_2d` en el Ejercicio 2).
- `predicciones_grilla = modelo.predict(puntos_grilla)`: le pide al modelo que prediga la clase para cada punto de esa grilla.
- `predicciones_grilla = predicciones_grilla.reshape(xx.shape)`: acomoda esas predicciones con la misma forma que la grilla, para poder pintarlas.
- `plt.contourf(xx, yy, predicciones_grilla, alpha=0.3)`: pinta el fondo del gráfico según la clase predicha en cada zona. `alpha=0.3` lo hace semi-transparente para que se vean los puntos reales encima.
- `plt.scatter(...)`: dibuja los puntos reales encima del fondo pintado, igual que en el Ejercicio 1.

### Consigna

Volvé a entrenar el modelo con `n_neighbors=1` y generá de nuevo este gráfico. ¿Cómo cambian los bordes de las zonas de color respecto a `n_neighbors=5`?

---

## Ejercicio 4 – Matriz de confusión

### Código

```python
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(modelo, X_test, y_test, display_labels=datos.target_names)
plt.title("Matriz de confusión - KNN")
plt.show()
```

### Explicación línea por línea

- `ConfusionMatrixDisplay.from_estimator(modelo, X_test, y_test, display_labels=datos.target_names)`: usa el modelo ya entrenado para predecir sobre `X_test`, compara contra `y_test`, y arma la matriz de confusión. `display_labels` pone los nombres reales de las especies en vez de 0, 1, 2.
- La diagonal de la matriz muestra los aciertos. Los valores fuera de la diagonal muestran en qué clase se confundió el modelo.

### Consigna

¿Hubo alguna confusión entre Versicolor y Virginica? Son las dos especies que más se parecen entre sí en este dataset.

---

## Ejercicio 5 – Precisión según el valor de k

### Código

```python
valores_k = range(1, 15)
scores = []

for k in valores_k:
    modelo_k = KNeighborsClassifier(n_neighbors=k)
    modelo_k.fit(X_train, y_train)
    scores.append(modelo_k.score(X_test, y_test))

plt.plot(valores_k, scores, marker="o")
plt.xlabel("Valor de k")
plt.ylabel("Precisión (accuracy)")
plt.title("Precisión según el valor de k")
plt.show()
```

### Explicación línea por línea

- `for k in valores_k:`: prueba distintos valores de k, del 1 al 14.
- En cada vuelta, entrena un modelo nuevo con ese k y guarda su precisión sobre `X_test` en la lista `scores`.
- `plt.plot(valores_k, scores, marker="o")`: grafica la precisión obtenida para cada valor de k. `marker="o"` marca cada punto con un círculo, para verlos mejor sobre la línea.

### Consigna

Mirando el gráfico, ¿qué valor de k te parece el mejor punto medio entre pocos vecinos (riesgo de overfitting) y muchos vecinos (riesgo de underfitting)?

---

## Cierre del laboratorio

Con estos 5 gráficos cubriste el ciclo completo de KNN: ver los datos antes de entrenar (scatter), entender cómo separa el modelo el espacio (frontera de decisión), evaluar sus aciertos y errores (matriz de confusión), y elegir el mejor hiperparámetro k (precisión vs k).
