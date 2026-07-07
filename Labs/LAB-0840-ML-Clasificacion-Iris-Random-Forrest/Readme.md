# LAB-0840 – ML: Clasificación Iris con Random Forest

## Objetivo

Entrenar un modelo Random Forest sobre el dataset Iris y comprender cómo un conjunto de árboles de decisión puede obtener mejores resultados que un único árbol mediante un sistema de votación.

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

## Consigna

¿Cuál de las tres especies parece más fácil de distinguir visualmente?

***

# Ejercicio 2 – Entrenar un árbol de decisión

Antes de usar Random Forest recordemos el modelo base.

## Código

```python
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X_2d = np.column_stack(
    (largo_petalo, ancho_petalo)
)

X_train, X_test, y_train, y_test = train_test_split(
    X_2d,
    y,
    test_size=0.3,
    random_state=42
)

arbol = DecisionTreeClassifier(
    random_state=42
)

arbol.fit(
    X_train,
    y_train
)

print(
    arbol.score(
        X_test,
        y_test
    )
)
```

## Explicación

Tenemos un único árbol tomando todas las decisiones.

## Consigna

Anotá el accuracy obtenido para compararlo más adelante.

***

# Ejercicio 3 – Entrenar Random Forest

## Código

```python
from sklearn.ensemble import RandomForestClassifier

bosque = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

bosque.fit(
    X_train,
    y_train
)

print(
    bosque.score(
        X_test,
        y_test
    )
)
```

## Explicación línea por línea

* `RandomForestClassifier()` crea muchos árboles.
* `n_estimators=100` indica que entrenará 100 árboles diferentes.
* Cada árbol ve una muestra ligeramente distinta de los datos.
* La predicción final surge de una votación.

### Concepto clave

Si los árboles votan:

```text
Setosa
Setosa
Setosa
Virginica
Setosa
Setosa
```

gana:

```text
Setosa
```

## Consigna

¿Mejoró el accuracy respecto al árbol individual?

***

# Ejercicio 4 – Frontera de decisión

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

predicciones = bosque.predict(
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

plt.title(
    "Frontera de decisión - Random Forest"
)

plt.show()
```

## Explicación

La frontera surge de la combinación de muchos árboles.

No pertenece a un árbol en particular.

Es el resultado de la votación de todo el bosque.

## Consigna

¿La frontera parece más estable que la del árbol individual?

***

# Ejercicio 5 – ¿Cuántos árboles hay?

## Código

```python
print(
    len(bosque.estimators_)
)
```

### Salida

```text
100
```

## Explicación

Cada elemento de:

```python
bosque.estimators_
```

es un árbol diferente.

## Consigna

¿Qué ventaja tendría usar muchos árboles en lugar de uno solo?

***

# Ejercicio 6 – Ver uno de los árboles

Aunque el bosque tenga 100 árboles, podemos inspeccionar uno.

## Código

```python
from sklearn.tree import plot_tree

plt.figure(figsize=(14,8))

plot_tree(
    bosque.estimators_[0],
    feature_names=[
        "Largo pétalo",
        "Ancho pétalo"
    ],
    class_names=datos.target_names,
    filled=True
)

plt.show()
```

## Explicación

* `estimators_[0]` es el primer árbol.
* Cada árbol del bosque es distinto.
* Ninguno toma exactamente las mismas decisiones.

## Consigna

¿Crees que este árbol individual tendrá el mismo accuracy que el bosque completo?

***

# Ejercicio 7 – Importancia de variables

Random Forest puede calcular qué variables fueron más útiles.

## Código

```python
import pandas as pd

importancias = pd.DataFrame(
    {
        "Feature": [
            "Largo pétalo",
            "Ancho pétalo"
        ],
        "Importancia": bosque.feature_importances_
    }
)

print(importancias)

plt.bar(
    importancias["Feature"],
    importancias["Importancia"]
)

plt.ylabel("Importancia")

plt.title(
    "Importancia de variables"
)

plt.show()
```

## Explicación

* Se analiza cuánto ayudó cada feature.
* Cuanto mayor sea el valor, más información aporta.

## Consigna

¿Cuál de las dos variables parece ser más importante?

***

# Ejercicio 8 – Matriz de confusión

## Código

```python
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(
    bosque,
    X_test,
    y_test,
    display_labels=datos.target_names
)

plt.title(
    "Matriz de confusión - Random Forest"
)

plt.show()
```

## Explicación

Permite comparar:

* clases reales
* clases predichas

## Consigna

¿Las confusiones se producen entre las mismas especies que en los modelos anteriores?

***

# Ejercicio 9 – Accuracy según cantidad de árboles

## Código

```python
cantidades = [
    1,
    5,
    10,
    20,
    50,
    100,
    200
]

scores = []

for cantidad in cantidades:

    modelo_tmp = RandomForestClassifier(
        n_estimators=cantidad,
        random_state=42
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
    cantidades,
    scores,
    marker="o"
)

plt.xlabel("Cantidad de árboles")
plt.ylabel("Accuracy")

plt.title(
    "Accuracy según cantidad de árboles"
)

plt.show()
```

## Explicación

Se entrena el bosque con distinta cantidad de árboles.

Observamos si agregar más árboles sigue aportando mejora o llega un punto donde la ganancia es mínima.

## Consigna

¿A partir de qué cantidad de árboles el accuracy deja prácticamente de mejorar?

***

# Cierre del laboratorio

Con este laboratorio aprendiste:

* Qué es un Random Forest.
* Cómo funciona la votación entre múltiples árboles.
* La diferencia entre un árbol individual y un ensamble.
* Cómo visualizar la frontera de decisión del bosque.
* Cómo inspeccionar árboles individuales.
* Cómo interpretar la importancia de variables.
* Cómo influye la cantidad de árboles en el resultado.

## Idea clave

Un Árbol de Decisión es como pedirle opinión a una sola persona.

Un Random Forest es como preguntarle a 100 expertos diferentes y quedarse con la respuesta que recibe más votos.

Por eso Random Forest suele ser uno de los primeros algoritmos de ensamble que se enseñan y uno de los más efectivos para datos tabulares del mundo real.
