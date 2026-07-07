# LAB-0820 – ML: Clasificación Iris con Logistic Regression

## Objetivo

Entrenar un modelo de Regresión Logística sobre el dataset Iris y comprender cómo un clasificador puede calcular probabilidades de pertenencia a cada especie antes de decidir la clase final.

***

# Requisitos previos

```bash
pip install scikit-learn matplotlib pandas
```

***

# El dataset

Iris viene incluido en scikit-learn.

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

```text
[[5.1 3.5 1.4 0.2]
 [4.9 3.0 1.4 0.2]
 [4.7 3.2 1.3 0.2]
 [4.6 3.1 1.5 0.2]
 [5.0 3.6 1.4 0.2]]

[0 0 0 0 0]

['setosa' 'versicolor' 'virginica']
```

* X contiene las features.
* y contiene la especie de cada flor.

***

# Ejercicio 1 – Scatter de los datos

Utilizaremos largo y ancho de pétalo.

## Código

```python
import matplotlib.pyplot as plt

largo_petalo = X[:, 2]
ancho_petalo = X[:, 3]

plt.scatter(
    largo_petalo,
    ancho_petalo,
    c=y
)

plt.xlabel("Largo de pétalo")
plt.ylabel("Ancho de pétalo")
plt.title("Iris - Scatter por especie")

plt.show()
```

## Explicación línea por línea

* `X[:,2]` obtiene el largo de pétalo.
* `X[:,3]` obtiene el ancho de pétalo.
* `c=y` colorea cada flor según su especie.

## Consigna

¿Parece posible separar las especies con una línea recta o la separación parece más compleja?

***

# Ejercicio 2 – Entrenar Logistic Regression

## Código

```python
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_2d = np.column_stack(
    (largo_petalo, ancho_petalo)
)

X_train, X_test, y_train, y_test = train_test_split(
    X_2d,
    y,
    test_size=0.3,
    random_state=42
)

modelo = LogisticRegression()

modelo.fit(
    X_train,
    y_train
)

print(modelo.score(X_test, y_test))
```

## Explicación línea por línea

* `LogisticRegression()` crea el clasificador.
* Aunque su nombre contiene la palabra "regresión", se utiliza para clasificación.
* Aprende los límites que separan las distintas clases.
* `score()` calcula el accuracy.

## Consigna

Compará el accuracy obtenido con el de KNN y Árbol de Decisión.

***

# Ejercicio 3 – Predicción de una flor nueva

## Código

```python
nueva_flor = [[4.5, 1.5]]

prediccion = modelo.predict(
    nueva_flor
)

print(prediccion)

print(
    datos.target_names[prediccion[0]]
)
```

### Salida posible

```text
[1]

versicolor
```

## Explicación línea por línea

* Creamos una flor imaginaria.
* Tiene largo de pétalo 4.5.
* Tiene ancho de pétalo 1.5.
* El modelo devuelve la clase más probable.

## Consigna

Probá otros valores y observá cómo cambia la predicción.

***

# Ejercicio 4 – Probabilidades por clase

Esta es la característica más importante de la Regresión Logística.

## Código

```python
probabilidades = modelo.predict_proba(
    nueva_flor
)

print(probabilidades)
```

### Salida posible

```text
[[0.01 0.85 0.14]]
```

## Explicación línea por línea

Las probabilidades corresponden a:

```text
[setosa, versicolor, virginica]
```

Por ejemplo:

```text
Setosa      : 1%
Versicolor  : 85%
Virginica   : 14%
```

El modelo elige la clase con la probabilidad más alta.

## Consigna

¿Por qué es más útil conocer las probabilidades que únicamente conocer la predicción final?

***

# Ejercicio 5 – Frontera de decisión

## Código

```python
minimo_x = largo_petalo.min() - 1
maximo_x = largo_petalo.max() + 1

minimo_y = ancho_petalo.min() - 1
maximo_y = ancho_petalo.max() + 1

xx, yy = np.meshgrid(
    np.arange(minimo_x, maximo_x, 0.1),
    np.arange(minimo_y, maximo_y, 0.1)
)

puntos_grilla = np.column_stack(
    (xx.ravel(), yy.ravel())
)

predicciones = modelo.predict(
    puntos_grilla
)

predicciones = predicciones.reshape(
    xx.shape
)

plt.contourf(
    xx,
    yy,
    predicciones,
    alpha=0.3
)

plt.scatter(
    largo_petalo,
    ancho_petalo,
    c=y,
    edgecolor="k"
)

plt.xlabel("Largo de pétalo")
plt.ylabel("Ancho de pétalo")
plt.title("Frontera de decisión - Logistic Regression")

plt.show()
```

## Explicación línea por línea

* Creamos una grilla de puntos.
* Predecimos la clase para cada punto.
* Pintamos el fondo según la predicción.
* Dibujamos las flores reales encima.

### Observación

A diferencia del Árbol de Decisión, las fronteras de Logistic Regression suelen ser aproximadamente rectas.

## Consigna

¿Las fronteras se parecen más a las de KNN o a las del Árbol?

***

# Ejercicio 6 – Matriz de confusión

## Código

```python
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(
    modelo,
    X_test,
    y_test,
    display_labels=datos.target_names
)

plt.title(
    "Matriz de confusión - Logistic Regression"
)

plt.show()
```

## Explicación línea por línea

* Se comparan las predicciones contra las clases reales.
* La diagonal muestra los aciertos.
* Los valores fuera de la diagonal muestran errores.

## Consigna

¿Entre qué especies aparecen las confusiones?

***

# Ejercicio 7 – Efecto de la regularización

La regularización evita que el modelo se ajuste demasiado a los datos.

## Código

```python
valores_C = [
    0.01,
    0.1,
    1,
    10,
    100
]

scores = []

for c in valores_C:

    modelo_tmp = LogisticRegression(
        C=c
    )

    modelo_tmp.fit(
        X_train,
        y_train
    )

    scores.append(
        modelo_tmp.score(
            X_test,
            y_test
        )
    )

plt.plot(
    valores_C,
    scores,
    marker="o"
)

plt.xscale("log")

plt.xlabel("Valor de C")
plt.ylabel("Accuracy")
plt.title("Accuracy según regularización")

plt.show()
```

## Explicación línea por línea

* `C` controla la regularización.
* Valores pequeños → más regularización.
* Valores grandes → menos regularización.
* Se comparan distintos modelos.

## Consigna

¿A partir de qué valor de C el accuracy deja de mejorar?

***

# Ejercicio 8 – Visualizar las probabilidades de una flor

## Código

```python
probs = modelo.predict_proba(
    nueva_flor
)[0]

plt.bar(
    datos.target_names,
    probs
)

plt.ylabel("Probabilidad")
plt.title("Probabilidad por especie")

plt.show()
```

## Explicación línea por línea

Este gráfico muestra cómo reparte la confianza el modelo entre las tres especies.

Un clasificador no siempre está completamente seguro.

## Consigna

Buscá una flor cuyos valores estén cerca de la frontera entre Versicolor y Virginica.

¿Las probabilidades aparecen más repartidas?

***

# Cierre del laboratorio

Con este laboratorio aprendiste a:

* Entrenar un clasificador Logistic Regression.
* Comprender por qué es un clasificador a pesar de llamarse "regresión".
* Obtener probabilidades para cada clase.
* Interpretar la confianza del modelo.
* Visualizar fronteras de decisión.
* Evaluar resultados mediante una matriz de confusión.
* Analizar el efecto de la regularización.

La principal ventaja de Logistic Regression es que además de predecir una clase, proporciona una probabilidad asociada a cada decisión, lo que la convierte en uno de los algoritmos más utilizados para problemas de clasificación en el mundo real.
