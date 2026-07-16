# LAB-0910 – AdvML: Comparación de Modelos de Clasificación

## Objetivo

Aplicar métricas de clasificación a varios modelos reales de Machine Learning usando el dataset **Breast Cancer Wisconsin**.

En este laboratorio vamos a comparar:

* KNN
* Árbol de Decisión
* Logistic Regression
* SVM
* Random Forest
* Gradient Boosting

utilizando:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC AUC
* Matriz de confusión
* Curvas ROC
* Tiempo de entrenamiento

La idea central es entender que:

> El mejor modelo no siempre es el que tiene mayor accuracy.

***

# Requisitos previos

```bash
pip install scikit-learn matplotlib pandas
```

***

# El dataset

Usaremos el dataset **Breast Cancer Wisconsin**, incluido en scikit-learn.

```python
from sklearn.datasets import load_breast_cancer

datos = load_breast_cancer()

X = datos.data
y = datos.target

print(X.shape)
print(y.shape)
print(datos.target_names)
print(datos.feature_names[:10])
```

### Salida esperada

```text
(569, 30)
(569,)
['malignant' 'benign']
['mean radius' 'mean texture' 'mean perimeter' ...]
```

## Explicación

* `X` contiene las variables predictoras.
* `y` contiene la clase real.
* Hay dos clases:
  * `0 = malignant`
  * `1 = benign`
* Es un problema de clasificación binaria.

## Consigna

¿Por qué este dataset permite discutir mejor las métricas que Iris?

***

# Ejercicio 1 – Crear un DataFrame para explorar los datos

## Código

```python
import pandas as pd

df = pd.DataFrame(
    X,
    columns=datos.feature_names
)

df["target"] = y

print(df.head())
print(df["target"].value_counts())
print(datos.target_names)
```

## Explicación línea por línea

* `pd.DataFrame(...)` convierte los datos en una tabla.
* `columns=datos.feature_names` asigna los nombres reales de las variables.
* `df["target"] = y` agrega la clase real.
* `value_counts()` cuenta cuántos casos hay de cada clase.

## Consigna

¿Hay la misma cantidad de casos benignos y malignos?

***

# Ejercicio 2 – Separar entrenamiento y prueba

## Código

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print(X_train.shape)
print(X_test.shape)
```

## Explicación línea por línea

* `test_size=0.3` deja el 30% de los datos para prueba.
* `random_state=42` hace que el resultado sea reproducible.
* `stratify=y` mantiene proporciones similares de clases en entrenamiento y prueba.

## Consigna

¿Por qué sería problemático que en el conjunto de prueba quedaran muy pocos casos malignant?

***

# Ejercicio 3 – Crear los modelos

## Código

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

modelos = {
    "KNN": KNeighborsClassifier(n_neighbors=5),

    "Árbol de Decisión": DecisionTreeClassifier(
        random_state=42
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=5000
    ),

    "SVM": SVC(
        probability=True,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
}
```

## Explicación

Creamos un diccionario de modelos para entrenarlos y evaluarlos todos de la misma manera.

### Nota importante

En SVM usamos:

```python
probability=True
```

porque después vamos a calcular probabilidades para ROC AUC.

## Consigna

¿Qué ventaja tiene guardar todos los modelos en un diccionario?

***

# Ejercicio 4 – Entrenar todos los modelos

## Código

```python
for nombre, modelo in modelos.items():

    modelo.fit(
        X_train,
        y_train
    )

    print(
        nombre,
        "entrenado"
    )
```

## Explicación línea por línea

* `for nombre, modelo in modelos.items()` recorre todos los modelos.
* `modelo.fit(...)` entrena cada modelo.
* Imprimimos el nombre para verificar que todos se entrenaron.

## Consigna

¿Cuál de estos modelos esperás que tarde más en entrenarse?

***

# Ejercicio 5 – Comparar métricas principales

## Código

```python
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

resultados = []

for nombre, modelo in modelos.items():

    y_pred = modelo.predict(X_test)

    y_proba = modelo.predict_proba(X_test)[:, 1]

    resultados.append(
        {
            "Modelo": nombre,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "ROC AUC": roc_auc_score(y_test, y_proba)
        }
    )

tabla_resultados = pd.DataFrame(resultados)

print(tabla_resultados)
```

## Explicación línea por línea

* `y_pred` contiene la clase predicha por el modelo.
* `y_proba` contiene la probabilidad de pertenecer a la clase positiva.
* Calculamos:
  * Accuracy
  * Precision
  * Recall
  * F1
  * ROC AUC
* Guardamos todo en una tabla.

## Consigna

¿El modelo con mayor Accuracy también tiene mayor Recall?

***

# Ejercicio 6 – Ordenar modelos por Accuracy

## Código

```python
tabla_accuracy = tabla_resultados.sort_values(
    by="Accuracy",
    ascending=False
)

print(tabla_accuracy)
```

## Explicación

Ordenamos los modelos desde el mayor accuracy al menor.

## Consigna

Si sólo miraras Accuracy, ¿qué modelo elegirías?

***

# Ejercicio 7 – Ordenar modelos por Recall

## Código

```python
tabla_recall = tabla_resultados.sort_values(
    by="Recall",
    ascending=False
)

print(tabla_recall)
```

## Explicación

Recall responde:

```text
De todos los casos reales positivos, ¿cuántos detecté?
```

En este dataset, el target positivo por defecto es `benign`, porque scikit-learn codifica:

```text
0 = malignant
1 = benign
```

Por eso hay que tener cuidado al interpretar Recall.

## Consigna

En un caso médico, ¿qué clase te interesaría detectar con mayor prioridad: benign o malignant?

***

# Ejercicio 8 – Cambiar el foco hacia malignant

Como `malignant` está codificado como `0`, vamos a calcular métricas considerando `malignant` como clase positiva.

## Código

```python
resultados_malignant = []

for nombre, modelo in modelos.items():

    y_pred = modelo.predict(X_test)

    resultados_malignant.append(
        {
            "Modelo": nombre,
            "Precision malignant": precision_score(
                y_test,
                y_pred,
                pos_label=0
            ),
            "Recall malignant": recall_score(
                y_test,
                y_pred,
                pos_label=0
            ),
            "F1 malignant": f1_score(
                y_test,
                y_pred,
                pos_label=0
            )
        }
    )

tabla_malignant = pd.DataFrame(resultados_malignant)

print(tabla_malignant)
```

## Explicación

Ahora cambiamos el foco del análisis.

En vez de preguntar:

```text
¿Qué tan bien detecto benign?
```

preguntamos:

```text
¿Qué tan bien detecto malignant?
```

## Consigna

¿Por qué esta tabla puede ser más importante que la tabla anterior?

***

# Ejercicio 9 – Classification Report

## Código

```python
from sklearn.metrics import classification_report

for nombre, modelo in modelos.items():

    y_pred = modelo.predict(X_test)

    print("\n==============================")
    print(nombre)
    print("==============================")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=datos.target_names
        )
    )
```

## Explicación

`classification_report` muestra:

* Precision por clase
* Recall por clase
* F1 por clase
* Support
* Macro average
* Weighted average

## Consigna

¿En qué modelo aparece mejor balance entre malignant y benign?

***

# Ejercicio 10 – Macro Average vs Weighted Average

## Concepto

En el `classification_report` aparecen dos promedios importantes.

## Macro Average

```text
Promedia las métricas dando el mismo peso a cada clase.
```

Es útil cuando nos importan todas las clases por igual.

## Weighted Average

```text
Promedia las métricas ponderando por la cantidad de casos de cada clase.
```

Las clases con más ejemplos pesan más.

## Consigna

En este problema médico, ¿mirarías más Macro Average o Weighted Average?

***

# Ejercicio 11 – Matriz de confusión de un modelo

Vamos a inspeccionar primero Gradient Boosting.

## Código

```python
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

modelo_elegido = modelos["Gradient Boosting"]

ConfusionMatrixDisplay.from_estimator(
    modelo_elegido,
    X_test,
    y_test,
    display_labels=datos.target_names
)

plt.title("Matriz de confusión - Gradient Boosting")
plt.show()
```

## Explicación

La matriz muestra:

```text
Predicciones correctas
Errores del modelo
Confusiones entre malignant y benign
```

## Consigna

¿Qué error considerás más grave?

```text
Predecir benign cuando era malignant
```

o

```text
Predecir malignant cuando era benign
```

***

# Ejercicio 12 – Matrices de confusión para todos los modelos

## Código

```python
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

fig, axes = plt.subplots(
    2,
    3,
    figsize=(15, 8)
)

axes = axes.ravel()

for i, (nombre, modelo) in enumerate(modelos.items()):

    ConfusionMatrixDisplay.from_estimator(
        modelo,
        X_test,
        y_test,
        display_labels=datos.target_names,
        ax=axes[i],
        colorbar=False
    )

    axes[i].set_title(nombre)

plt.tight_layout()
plt.show()
```

## Explicación

Este gráfico permite comparar visualmente los errores de todos los modelos.

## Consigna

¿Todos los modelos se equivocan en los mismos casos o algunos cometen errores distintos?

***

# Ejercicio 13 – Curvas ROC para todos los modelos

## Código

```python
from sklearn.metrics import RocCurveDisplay

plt.figure(figsize=(8, 6))

for nombre, modelo in modelos.items():

    RocCurveDisplay.from_estimator(
        modelo,
        X_test,
        y_test,
        name=nombre
    )

plt.title("Curvas ROC - Comparación de modelos")
plt.show()
```

## Explicación

La curva ROC muestra cómo cambia el rendimiento del modelo cuando movemos el umbral de decisión.

En lugar de evaluar una única decisión fija, evalúa el comportamiento del modelo en varios umbrales.

## Consigna

¿Qué modelo tiene una curva más cercana a la esquina superior izquierda?

***

# Ejercicio 14 – Ranking por ROC AUC

## Código

```python
tabla_auc = tabla_resultados.sort_values(
    by="ROC AUC",
    ascending=False
)

print(tabla_auc[["Modelo", "ROC AUC"]])
```

## Explicación

ROC AUC resume la curva ROC en un número.

Valores cercanos a 1 indican mejor capacidad de separación.

## Consigna

¿El mejor modelo por ROC AUC coincide con el mejor modelo por Accuracy?

***

# Ejercicio 15 – Medir tiempo de entrenamiento

## Código

```python
import time

tiempos = []

for nombre, modelo in modelos.items():

    inicio = time.time()

    modelo.fit(
        X_train,
        y_train
    )

    fin = time.time()

    tiempos.append(
        {
            "Modelo": nombre,
            "Tiempo entrenamiento": fin - inicio
        }
    )

tabla_tiempos = pd.DataFrame(tiempos)

print(tabla_tiempos)
```

## Explicación

Además de las métricas predictivas, también puede importar:

* velocidad
* coste computacional
* complejidad
* facilidad de mantenimiento

## Consigna

¿Elegirías un modelo mucho más lento si sólo mejora mínimamente el F1 Score?

***

# Ejercicio 16 – Gráfico comparativo de métricas

## Código

```python
tabla_resultados.plot(
    x="Modelo",
    y=[
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC AUC"
    ],
    kind="bar",
    figsize=(14, 7)
)

plt.ylim(0.8, 1.0)
plt.ylabel("Valor")
plt.title("Comparación de métricas por modelo")
plt.xticks(rotation=45)
plt.legend(loc="lower right")

plt.show()
```

## Explicación

Este gráfico permite comparar de forma visual todos los modelos y métricas.

## Consigna

¿Qué modelo parece más equilibrado?

***

# Ejercicio 17 – Tabla final combinada

## Código

```python
tabla_final = tabla_resultados.merge(
    tabla_malignant,
    on="Modelo"
)

tabla_final = tabla_final.merge(
    tabla_tiempos,
    on="Modelo"
)

print(tabla_final)
```

## Explicación

La tabla final incluye:

* Métricas generales
* Métricas enfocadas en malignant
* Tiempo de entrenamiento

## Consigna

¿Cuál modelo elegirías si tu prioridad fuera detectar la mayor cantidad posible de casos malignant?

***

# Ejercicio 18 – Elegir el modelo según el contexto

## Caso 1 – Diagnóstico médico

Prioridad:

```text
No dejar pasar casos malignant.
```

Métrica más importante:

```text
Recall malignant
```

***

## Caso 2 – Sistema con revisión humana

Prioridad:

```text
Reducir falsas alarmas.
```

Métrica más importante:

```text
Precision
```

***

## Caso 3 – Modelo general equilibrado

Prioridad:

```text
Buen equilibrio entre errores.
```

Métrica más importante:

```text
F1 Score
```

***

## Caso 4 – Comparación global de capacidad discriminativa

Prioridad:

```text
Separar bien las clases en distintos umbrales.
```

Métrica más importante:

```text
ROC AUC
```

***

# Cierre del laboratorio

Con este laboratorio aprendiste a:

* Entrenar varios modelos sobre el mismo dataset.
* Comparar clasificadores con métricas reales.
* Entender por qué Accuracy no alcanza.
* Interpretar Precision, Recall y F1 en contexto.
* Usar ROC AUC para comparar capacidad de separación.
* Analizar matrices de confusión.
* Diferenciar métricas generales de métricas orientadas a una clase específica.
* Considerar el tiempo de entrenamiento como parte de la decisión.
* Elegir modelos según el coste real del error.

***

# Idea clave

> Comparar modelos no es mirar una única métrica. Es entender qué error importa más para el problema que estamos resolviendo.

En este laboratorio, si pensamos en diagnóstico médico, la pregunta principal no es:

```text
¿Qué modelo tiene mayor Accuracy?
```

sino:

```text
¿Qué modelo reduce más el riesgo de no detectar un caso malignant?
```

Ese es el puente perfecto entre métricas técnicas y decisiones reales.
