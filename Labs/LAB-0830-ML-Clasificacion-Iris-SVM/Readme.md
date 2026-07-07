# LAB-0830 – ML: Clasificación Iris con Support Vector Machine (SVM)

## Objetivo

Entrenar un modelo SVM sobre el dataset Iris y comprender el concepto de hiperplano separador, margen máximo y vectores soporte.

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

***

# Ejercicio 1 – Visualizando las flores

Trabajaremos con:

* Largo de pétalo
* Ancho de pétalo

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

* Cada punto representa una flor.
* Los colores indican la especie.
* Estamos observando únicamente dos features para poder dibujar los datos en 2 dimensiones.

## Consigna

¿Parece posible separar las especies mediante una o varias líneas rectas?

***

# Ejercicio 2 – Entrenar el modelo SVM

## Código

```python
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X_2d = np.column_stack(
    (largo_petalo, ancho_petalo)
)

X_train, X_test, y_train, y_test = train_test_split(
    X_2d,
    y,
    test_size=0.3,
    random_state=42
)

modelo = SVC(
    kernel="linear"
)

modelo.fit(
    X_train,
    y_train
)

print(
    modelo.score(
        X_test,
        y_test
    )
)
```

## Explicación línea por línea

* `SVC` significa Support Vector Classifier.
* `kernel="linear"` indica que buscamos una separación mediante líneas rectas.
* El modelo intenta construir el hiperplano que mejor separa las clases.

## Consigna

Compará el accuracy obtenido contra:

* KNN
* Árbol de Decisión
* Logistic Regression

***

# Ejercicio 3 – Frontera de decisión

## Código

```python
minimo_x = largo_petalo.min() - 1
maximo_x = largo_petalo.max() + 1

minimo_y = ancho_petalo.min() - 1
maximo_y = ancho_petalo.max() + 1

xx, yy = np.meshgrid(
    np.arange(minimo_x, maximo_x, 0.05),
    np.arange(minimo_y, maximo_y, 0.05)
)

puntos = np.column_stack(
    (xx.ravel(), yy.ravel())
)

predicciones = modelo.predict(
    puntos
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

plt.title("Frontera de decisión - SVM")

plt.show()
```

## Explicación línea por línea

El fondo muestra la especie que el modelo predice en cada zona del espacio.

A diferencia del Árbol de Decisión:

* Las fronteras son rectas.
* Las regiones son más limpias.
* El objetivo es maximizar la separación entre clases.

## Consigna

¿Las fronteras se parecen más a Logistic Regression o a Árboles de Decisión?

***

# Ejercicio 4 – Dibujar los vectores soporte

Los vectores soporte son los puntos que determinan dónde queda la frontera.

## Código

```python
plt.figure(figsize=(8,6))

plt.scatter(
    largo_petalo,
    ancho_petalo,
    c=y
)

plt.scatter(
    modelo.support_vectors_[:,0],
    modelo.support_vectors_[:,1],
    s=200,
    facecolors="none",
    edgecolors="red"
)

plt.xlabel("Largo de pétalo")
plt.ylabel("Ancho de pétalo")

plt.title("Vectores soporte")

plt.show()
```

## Explicación línea por línea

* `support_vectors_` contiene las observaciones más importantes.
* Son las flores más cercanas a la frontera.
* Si esas flores desaparecieran, la frontera probablemente cambiaría.

### Concepto clave

SVM no utiliza todos los puntos para definir la frontera.

Sólo necesita unos pocos puntos críticos:

**los vectores soporte.**

## Consigna

¿Por qué creés que esos puntos son los más importantes para el modelo?

***

# Ejercicio 5 – Matriz de confusión

## Código

```python
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(
    modelo,
    X_test,
    y_test,
    display_labels=datos.target_names
)

plt.title("Matriz de confusión - SVM")

plt.show()
```

## Explicación línea por línea

La diagonal muestra aciertos.

Los elementos fuera de la diagonal muestran errores.

## Consigna

¿Las confusiones aparecen entre las mismas especies que en Logistic Regression?

***

# Ejercicio 6 – Analizando el margen

Una de las ideas fundamentales de SVM es maximizar el margen.

## Código

```python
print(
    "Cantidad de vectores soporte:"
)

print(
    modelo.n_support_
)
```

### Salida posible

```text
[2 7 8]
```

## Explicación línea por línea

El resultado indica cuántos vectores soporte utiliza cada clase.

Una buena separación suele requerir pocos vectores soporte.

## Consigna

¿Te sorprende que el modelo necesite tan pocos puntos para construir su frontera?

***

# Ejercicio 7 – El parámetro C

C controla cuánto castiga el modelo los errores.

## Código

```python
valores_c = [
    0.01,
    0.1,
    1,
    10,
    100
]

scores = []

for c in valores_c:

    modelo_tmp = SVC(
        kernel="linear",
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
    valores_c,
    scores,
    marker="o"
)

plt.xscale("log")

plt.xlabel("Valor de C")
plt.ylabel("Accuracy")

plt.title(
    "Accuracy según valor de C"
)

plt.show()
```

## Explicación línea por línea

* C pequeño → margen más amplio.
* C grande → menos tolerancia a errores.
* El modelo busca un equilibrio entre ambas cosas.

## Consigna

¿A partir de qué valor de C el accuracy deja de mejorar?

***

# Ejercicio 8 – SVM lineal vs SVM radial (RBF)

Ahora compararemos dos kernels distintos.

## Código

```python
modelo_lineal = SVC(
    kernel="linear"
)

modelo_rbf = SVC(
    kernel="rbf"
)

modelo_lineal.fit(
    X_train,
    y_train
)

modelo_rbf.fit(
    X_train,
    y_train
)

print(
    "Accuracy Lineal:",
    modelo_lineal.score(
        X_test,
        y_test
    )
)

print(
    "Accuracy RBF:",
    modelo_rbf.score(
        X_test,
        y_test
    )
)
```

## Explicación línea por línea

* Kernel lineal → fronteras rectas.
* Kernel RBF → fronteras curvas.
* RBF permite separar problemas más complejos.

## Consigna

¿Hay diferencia de accuracy entre ambos kernels en el dataset Iris?

¿Por qué creés que ocurre?

***

# Cierre del laboratorio

Con este laboratorio aprendiste:

* Qué es un SVM.
* El concepto de hiperplano separador.
* Qué significa maximizar el margen.
* Qué son los vectores soporte.
* Cómo visualizar las fronteras de decisión.
* Cómo evaluar el modelo mediante una matriz de confusión.
* Cómo influye el parámetro C.
* La diferencia entre un kernel lineal y uno no lineal.

La gran idea detrás de SVM es que no busca simplemente separar las clases, sino encontrar la separación más robusta posible, utilizando únicamente los ejemplos más importantes: los vectores soporte. Esto lo convirtió durante muchos años en uno de los algoritmos más utilizados para problemas de clasificación tabular antes de la popularización del Deep Learning.
