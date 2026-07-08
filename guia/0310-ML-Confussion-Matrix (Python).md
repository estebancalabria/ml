# Clase: Matriz de Confusión (Confusion Matrix) — Anexo con Python

## Objetivos

Al finalizar esta clase los estudiantes serán capaces de:

* Comprender por qué es necesario evaluar modelos de clasificación.
* Entender cómo se comparan distintos algoritmos de Machine Learning.
* Comprender qué representa una matriz de confusión.
* Identificar True Positives, True Negatives, False Positives y False Negatives.
* Interpretar correctamente una matriz de confusión.
* Comprender el significado de la diagonal principal.
* Comparar distintos algoritmos utilizando matrices de confusión.
* Comprender matrices binarias y multiclase.
* Entender que la matriz de confusión sirve como base para métricas más avanzadas.

## Secuencia

```text
Tenemos varios algoritmos
        ↓
¿Cómo elegir el mejor?
        ↓
Training Data
        ↓
Testing Data
        ↓
Predicciones
        ↓
Aciertos y errores
        ↓
Matriz de Confusión
        ↓
True Positive
True Negative
False Positive
False Negative
        ↓
Interpretar resultados
        ↓
Comparar modelos
        ↓
Limitaciones
        ↓
Matrices multiclase
        ↓
Matrices NxN
        ↓
Precision y Recall
        ↓
Conclusión
```

***

# Preparación Inicial

Durante toda la clase utilizaremos el dataset Breast Cancer incluido en Scikit-Learn.

## Ejecutar

```python
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)

print("Pacientes:", len(X))
print("Variables :", X.shape[1])
```

***

## ¿Qué representan estos datos?

```text
X = Mediciones clínicas

y = Diagnóstico

0 = maligno
1 = benigno
```

***

# Tenemos Varios Algoritmos

Supongamos que queremos construir un sistema que ayude a detectar enfermedades.

Podemos utilizar distintos algoritmos:

* Regresión Logística
* KNN
* Random Forest

***

## Pregunta

¿Cuál debemos elegir?

```text
Logistic Regression

KNN

Random Forest
```

Todos parecen razonables.

¿Cómo sabemos cuál funciona mejor?

***

# Entrenamiento y Evaluación

Como vimos anteriormente, dividimos nuestros datos.

## Ejecutar

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

print("Train:", X_train.shape)
print("Test :", X_test.shape)
```

***

## Recordatorio

```text
Training Data
```

Sirve para aprender.

***

```text
Testing Data
```

Sirve para evaluar.

***

# Entrenar un Modelo

Comencemos con Random Forest.

## Ejecutar

```python
from sklearn.ensemble import RandomForestClassifier

modelo = RandomForestClassifier(
    random_state=42
)

modelo.fit(
    X_train,
    y_train
)
```

***

# Realizar Predicciones

Ahora utilizamos pacientes que el modelo nunca vio.

## Ejecutar

```python
predicciones = modelo.predict(X_test)

print(predicciones[:10])
```

***

## ¿Qué obtenemos?

Para cada paciente:

```text
0 = maligno

1 = benigno
```

***

# Comparar Predicción y Realidad

Veamos algunos ejemplos.

## Ejecutar

```python
for i in range(10):

    print(
        "Real:",
        y_test[i],
        "Predicción:",
        predicciones[i]
    )
```

***

Cada predicción puede ser:

```text
Correcta
```

o

```text
Incorrecta
```

***

# Necesitamos Resumir Resultados

Si tenemos cientos de pacientes no podemos revisar uno por uno.

Necesitamos una forma compacta de resumir:

* Aciertos
* Errores

Para eso utilizamos la:

# Matriz de Confusión

***

# Creando una Matriz de Confusión

## Ejecutar

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    y_test,
    predicciones
)

print(cm)
```

***

Ejemplo de salida:

```text
[[50  4]
 [ 3 86]]
```

***

# Interpretación

```text
[[50  4]
 [ 3 86]]
```

Representa:

```text
                REAL

           0          1

PREDICE 0  50         4

PREDICE 1   3        86
```

***

# Aciertos

Los valores de la diagonal principal son:

```text
50

86
```

***

Representan:

```text
Clasificaciones correctas
```

***

# Errores

Los valores fuera de la diagonal son:

```text
4

3
```

***

Representan:

```text
Clasificaciones incorrectas
```

***

# Regla Fundamental

```text
Diagonal = Correcto
```

***

```text
Fuera de la diagonal = Error
```

***

# Visualizar la Matriz

Una imagen suele resultar más fácil de interpretar.

## Ejecutar

```python
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay.from_predictions(
    y_test,
    predicciones
)

plt.show()
```

***

## Interpretación

Las celdas más oscuras suelen indicar mayor cantidad de casos.

Idealmente queremos observar:

```text
Muchos valores en la diagonal
```

y

```text
Pocos valores fuera de ella
```

***

# Comparando Modelos

Ahora comparemos varios algoritmos.

***

## Regresión Logística

```python
from sklearn.linear_model import LogisticRegression

logistic = LogisticRegression(
    max_iter=10000
)

logistic.fit(
    X_train,
    y_train
)

pred_logistic = logistic.predict(X_test)

print(
    confusion_matrix(
        y_test,
        pred_logistic
    )
)
```

***

## KNN

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(
    X_train,
    y_train
)

pred_knn = knn.predict(X_test)

print(
    confusion_matrix(
        y_test,
        pred_knn
    )
)
```

***

## Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    random_state=42
)

rf.fit(
    X_train,
    y_train
)

pred_rf = rf.predict(X_test)

print(
    confusion_matrix(
        y_test,
        pred_rf
    )
)
```

***

# Pregunta

Observa las tres matrices.

¿Cuál tiene:

```text
Más valores en la diagonal?
```

***

¿Y cuál tiene:

```text
Menos errores?
```

***

# Accuracy

Podemos calcular una métrica básica.

## Ejecutar

```python
from sklearn.metrics import accuracy_score

print(
    accuracy_score(
        y_test,
        pred_rf
    )
)
```

***

Interpretación:

```text
Accuracy =
Porcentaje de aciertos
```

***

# Ejemplo Intuitivo

Supongamos que una matriz es:

```text
[[90 10]
 [20 80]]
```

***

Los aciertos son:

```text
90 + 80 = 170
```

***

Los errores son:

```text
10 + 20 = 30
```

***

Total:

```text
200 observaciones
```

***

Accuracy:

```text
170 / 200 = 0.85
```

***

```text
85%
```

***

# ¿Qué Significan los Cuatro Casos?

Consideremos una enfermedad.

Las posibles situaciones son:

***

## True Positive

```text
Paciente enfermo

Predicción: Enfermo
```

Correcto.

***

## True Negative

```text
Paciente sano

Predicción: Sano
```

Correcto.

***

## False Positive

```text
Paciente sano

Predicción: Enfermo
```

Error.

***

## False Negative

```text
Paciente enfermo

Predicción: Sano
```

Error.

***

# Extrayendo TP, TN, FP y FN

## Ejecutar

```python
tn, fp, fn, tp = confusion_matrix(
    y_test,
    pred_rf
).ravel()

print("TN =", tn)
print("FP =", fp)
print("FN =", fn)
print("TP =", tp)
```

***

# ¿Todos los Problemas Son Binarios?

No.

Hasta ahora teníamos:

```text
Maligno

Benigno
```

Dos clases.

***

Esto produce una matriz:

```text
2 x 2
```

***

# Problemas Multiclase

Imaginemos ahora que queremos clasificar flores.

Las posibles categorías son:

```text
Setosa

Versicolor

Virginica
```

***

Ahora tenemos:

```text
3 clases
```

***

# Cargar Dataset Iris

## Ejecutar

```python
from sklearn.datasets import load_iris

X, y = load_iris(
    return_X_y=True
)

print("Clases:", set(y))
```

***

# Entrenar un Clasificador

## Ejecutar

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=42
)

modelo = DecisionTreeClassifier()

modelo.fit(
    X_train,
    y_train
)
```

***

# Crear Matriz Multiclase

## Ejecutar

```python
predicciones = modelo.predict(X_test)

cm = confusion_matrix(
    y_test,
    predicciones
)

print(cm)
```

***

Ejemplo:

```text
[[15  0  0]
 [ 0 11  1]
 [ 0  0 11]]
```

***

# Interpretación

Ahora tenemos:

```text
3 filas

3 columnas
```

***

Cada fila corresponde a una predicción.

Cada columna corresponde a una clase real.

***

La regla sigue siendo exactamente la misma.

```text
Diagonal = Correcto
```

***

```text
Fuera de la diagonal = Error
```

***

# Visualizando la Matriz Multiclase

## Ejecutar

```python
ConfusionMatrixDisplay.from_predictions(
    y_test,
    predicciones
)

plt.show()
```

***

# Regla General

Si tenemos:

```text
2 clases
```

obtenemos:

```text
2 x 2
```

***

Si tenemos:

```text
3 clases
```

obtenemos:

```text
3 x 3
```

***

Si tenemos:

```text
10 clases
```

obtenemos:

```text
10 x 10
```

***

Si tenemos:

```text
N clases
```

obtenemos:

```text
N x N
```

***

# ¿Qué Viene Después?

La matriz de confusión es el punto de partida para métricas como:

* Precision
* Recall
* F1 Score
* Sensitivity
* Specificity
* ROC AUC

Todas ellas utilizan información obtenida de la matriz de confusión.

***

# Conceptos de Python Aprendidos

Durante esta clase utilizamos:

```python
predict()
```

Para realizar predicciones.

***

```python
confusion_matrix()
```

Para construir matrices de confusión.

***

```python
ConfusionMatrixDisplay
```

Para visualizar matrices.

***

```python
accuracy_score()
```

Para calcular accuracy.

***

```python
ravel()
```

Para obtener:

```text
TN
FP
FN
TP
```

***

# Resumen Final

La matriz de confusión permite resumir los resultados de un clasificador.

Las filas representan:

```text
Predicciones
```

***

Las columnas representan:

```text
Valores reales
```

***

La diagonal contiene:

```text
Aciertos
```

***

Fuera de la diagonal encontramos:

```text
Errores
```

***

Las matrices pueden utilizarse tanto para:

```text
Clasificación binaria
```

como para:

```text
Clasificación multiclase
```

***

# Mensaje de cierre

> Una matriz de confusión transforma cientos o miles de predicciones en una única representación visual que permite entender rápidamente dónde acierta y dónde se equivoca un modelo de Machine Learning. 🚀
