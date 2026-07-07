Excelente. Para mantener exactamente el mismo estilo pedagógico que el LAB-0800 de KNN, el siguiente laboratorio debería aprovechar las fortalezas visuales del Árbol de Decisión:

1. Scatter de los datos (igual que KNN).
2. Entrenar el árbol.
3. Dibujar el árbol (la gran ventaja didáctica).
4. Frontera de decisión.
5. Matriz de confusión.
6. Importancia de variables.

Eso permite explicar perfectamente cómo el modelo "pregunta cosas" sobre largo y ancho de pétalo para tomar decisiones.

***

# LAB-0810 – ML: Iris con Árbol de Decisión

## Objetivo

Entrenar un modelo de Árbol de Decisión sobre el dataset Iris y analizar cómo toma decisiones observando el árbol generado, la frontera de decisión, la matriz de confusión y la importancia de las variables.

## Requisitos previos

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

* X contiene las 4 features.
* y contiene la clase de cada flor.

***

# Ejercicio 1 – Scatter de dos features

Usaremos largo y ancho de pétalo.

## Código

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

## Explicación línea por línea

* `X[:,2]` obtiene la columna "largo de pétalo".
* `X[:,3]` obtiene la columna "ancho de pétalo".
* `plt.scatter(...)` grafica una flor por punto.
* `c=y` colorea por especie.

## Consigna

¿Cuál de las especies parece más fácil de separar visualmente?

***

# Ejercicio 2 – Entrenar el árbol

## Código

```python
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X_2d = np.column_stack((largo_petalo, ancho_petalo))

X_train, X_test, y_train, y_test = train_test_split(
    X_2d,
    y,
    test_size=0.3,
    random_state=42
)

modelo = DecisionTreeClassifier(random_state=42)

modelo.fit(X_train, y_train)

print(modelo.score(X_test, y_test))
```

## Explicación línea por línea

* `DecisionTreeClassifier()` crea un árbol de clasificación.
* `fit()` construye el árbol buscando preguntas que separen mejor las especies.
* `score()` calcula el accuracy sobre datos nunca vistos.

## Consigna

Volvé a ejecutar cambiando:

```python
DecisionTreeClassifier(max_depth=2)
```

¿Qué ocurre con el accuracy?

***

# Ejercicio 3 – Dibujar el árbol

Este es el gráfico más característico de este algoritmo.

## Código

```python
from sklearn.tree import plot_tree

plt.figure(figsize=(12,8))

plot_tree(
    modelo,
    feature_names=[
        "Largo pétalo",
        "Ancho pétalo"
    ],
    class_names=datos.target_names,
    filled=True
)

plt.show()
```

## Explicación línea por línea

* Cada rectángulo representa una decisión.
* El árbol evalúa preguntas del tipo:

```text
¿Ancho de pétalo <= 0.8?
```

o

```text
¿Largo de pétalo <= 2.4?
```

* Las ramas izquierdas corresponden a "Sí".
* Las ramas derechas corresponden a "No".
* Los nodos finales muestran la clase predicha.

## Consigna

¿Cuál es la primera pregunta que realiza el árbol?

¿Por qué creés que eligió esa feature?

***

# Ejercicio 4 – Frontera de decisión

## Código

```python
minimo_x, maximo_x = largo_petalo.min() - 1, largo_petalo.max() + 1
minimo_y, maximo_y = ancho_petalo.min() - 1, ancho_petalo.max() + 1

xx, yy = np.meshgrid(
    np.arange(minimo_x, maximo_x, 0.1),
    np.arange(minimo_y, maximo_y, 0.1)
)

puntos_grilla = np.column_stack(
    (xx.ravel(), yy.ravel())
)

predicciones = modelo.predict(puntos_grilla)
predicciones = predicciones.reshape(xx.shape)

plt.contourf(xx, yy, predicciones, alpha=0.3)

plt.scatter(
    largo_petalo,
    ancho_petalo,
    c=y,
    edgecolor="k"
)

plt.xlabel("Largo de pétalo")
plt.ylabel("Ancho de pétalo")
plt.title("Frontera de decisión - Árbol")

plt.show()
```

## Explicación línea por línea

* Se genera una grilla de puntos que cubre todo el gráfico.
* Se predice una clase para cada punto.
* Se colorea cada zona según la clase predicha.
* Los puntos reales se dibujan encima.

### Observación importante

En los árboles las fronteras suelen ser rectangulares porque las decisiones son preguntas sobre una sola variable cada vez.

## Consigna

¿Las fronteras se parecen a las de KNN o son más geométricas?

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

plt.title("Matriz de confusión - Árbol")
plt.show()
```

## Explicación línea por línea

La diagonal contiene los aciertos.

Los valores fuera de la diagonal indican errores de clasificación.

## Consigna

¿Entre qué especies aparecen los errores?

***

# Ejercicio 6 – Importancia de variables

Los árboles permiten ver qué variables fueron más importantes para tomar decisiones.

## Código

```python
import pandas as pd

importancias = pd.DataFrame(
    {
        "Feature": [
            "Largo pétalo",
            "Ancho pétalo"
        ],
        "Importancia": modelo.feature_importances_
    }
)

print(importancias)

plt.bar(
    importancias["Feature"],
    importancias["Importancia"]
)

plt.title("Importancia de variables")
plt.ylabel("Importancia")

plt.show()
```

## Explicación línea por línea

* `feature_importances_` es una propiedad del árbol.
* Indica cuánto ayudó cada variable a reducir la incertidumbre.
* Cuanto mayor sea el valor, más importante fue esa feature.

## Consigna

¿Cuál de las dos variables fue más importante para clasificar las flores?

***

# Ejercicio 7 – Profundidad del árbol

Vamos a comparar árboles de distinta complejidad.

## Código

```python
profundidades = range(1, 11)
scores = []

for profundidad in profundidades:

    modelo_tmp = DecisionTreeClassifier(
        max_depth=profundidad,
        random_state=42
    )

    modelo_tmp.fit(X_train, y_train)

    scores.append(
        modelo_tmp.score(X_test, y_test)
    )

plt.plot(
    profundidades,
    scores,
    marker="o"
)

plt.xlabel("Profundidad máxima")
plt.ylabel("Accuracy")

plt.title("Accuracy según profundidad")

plt.show()
```

## Explicación línea por línea

* Se entrenan varios árboles.
* Cada uno tiene una profundidad máxima distinta.
* Se mide su accuracy.
* Se observa cuándo el árbol deja de mejorar.

## Consigna

¿A partir de qué profundidad el accuracy deja prácticamente de aumentar?

***

# Cierre del laboratorio

Con este laboratorio cubriste el flujo completo de un Árbol de Decisión:

* Visualizar los datos originales.
* Entrenar el modelo.
* Entender cómo toma decisiones observando el árbol.
* Analizar las zonas que clasifica mediante la frontera de decisión.
* Evaluar errores mediante la matriz de confusión.
* Identificar las variables más importantes.
* Controlar el overfitting ajustando la profundidad del árbol.

La ventaja principal de los Árboles de Decisión es que son uno de los modelos de Machine Learning más fáciles de explicar, ya que cada predicción puede seguirse recorriendo las preguntas del árbol desde la raíz hasta una hoja final.
