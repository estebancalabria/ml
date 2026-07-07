# LAB-0900 – ML: Entendiendo las Métricas de Clasificación

## Objetivo

Comprender las métricas más importantes de clasificación:

* Accuracy
* Matriz de confusión
* Precision
* Recall
* F1 Score
* ROC Curve
* ROC AUC
* Macro Average
* Weighted Average
* Comparación de modelos

**Sin entrenar modelos.**

Trabajaremos únicamente con:

```python
y_real
y_pred
```

para entender qué mide cada métrica y cuándo conviene utilizarla.

***

# Requisitos previos

```bash
pip install scikit-learn matplotlib pandas seaborn
```

***

# Introducción

Cuando entrenamos un modelo de Machine Learning, el objetivo no es únicamente obtener predicciones.

También debemos responder:

```text
¿Es bueno el modelo?
```

Para ello utilizamos métricas.

En este laboratorio vamos a aprenderlas antes de entrenar algoritmos reales.

***

# Ejercicio 1 – Accuracy

Supongamos que tenemos un clasificador médico.

## Datos

```python
y_real = [
    1,1,1,1,1,
    0,0,0,0,0
]

y_pred = [
    1,1,1,1,0,
    0,0,0,0,0
]
```

Donde:

```text
1 = Enfermo
0 = Sano
```

***

## Código

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(
    y_real,
    y_pred
)

print(accuracy)
```

### Salida

```text
0.9
```

***

## Explicación línea por línea

Hay:

```text
10 pacientes
```

y el modelo acierta:

```text
9
```

Por tanto:

```text
Accuracy = 9 / 10 = 0.90
```

***

## Consigna

Calcula el accuracy manualmente antes de ejecutar el código.

***

# Ejercicio 2 – Matriz de confusión

Veamos exactamente dónde se equivoca el modelo.

## Código

```python
from sklearn.metrics import confusion_matrix

matriz = confusion_matrix(
    y_real,
    y_pred
)

print(matriz)
```

### Salida

```text
[[5 0]
 [1 4]]
```

***

## Explicación visual

```text
                 Predicho

               Sano   Enfermo

Real Sano        5       0

Real Enfermo     1       4
```

***

## Conceptos

### True Positive (TP)

```text
Enfermo y predijo enfermo
```

### True Negative (TN)

```text
Sano y predijo sano
```

### False Positive (FP)

```text
Sano pero predijo enfermo
```

### False Negative (FN)

```text
Enfermo pero predijo sano
```

***

## Consigna

Identifica:

```text
TP
TN
FP
FN
```

***

# Ejercicio 3 – Precision

Ahora nos interesa responder:

> De todo lo que predije como enfermo, ¿cuánto era realmente enfermo?

## Datos

```python
y_real = [
    1,1,1,1,
    0,0,0,0
]

y_pred = [
    1,1,1,1,
    1,1,0,0
]
```

***

## Código

```python
from sklearn.metrics import precision_score

precision = precision_score(
    y_real,
    y_pred
)

print(precision)
```

***

## Explicación

El modelo predijo:

```text
6 enfermos
```

pero sólo:

```text
4
```

eran realmente enfermos.

Por tanto:

```text
Precision = 4 / 6
```

***

## Consigna

¿Por qué Precision es importante en un filtro de spam?

***

# Ejercicio 4 – Recall

Ahora queremos responder:

> De todos los enfermos reales, ¿cuántos encontré?

## Datos

```python
y_real = [
    1,1,1,1,1,
    0,0,0,0,0
]

y_pred = [
    1,1,0,0,0,
    0,0,0,0,0
]
```

***

## Código

```python
from sklearn.metrics import recall_score

recall = recall_score(
    y_real,
    y_pred
)

print(recall)
```

***

## Explicación

Había:

```text
5 enfermos
```

El modelo detectó:

```text
2
```

Por tanto:

```text
Recall = 2 / 5
```

***

## Consigna

¿Por qué Recall suele ser especialmente importante en medicina?

***

# Ejercicio 5 – Cuando Accuracy engaña

Este es el ejercicio más importante del laboratorio.

## Situación

Tenemos:

```text
1000 pacientes
```

De ellos:

```text
990 sanos
10 enfermos
```

***

## Modelo absurdo

Predice:

```text
Sano para todos
```

***

## Código

```python
y_real = [1]*10 + [0]*990

y_pred = [0]*1000

from sklearn.metrics import accuracy_score

print(
    accuracy_score(
        y_real,
        y_pred
    )
)
```

### Salida

```text
0.99
```

***

## Explicación

El accuracy es:

```text
99%
```

pero el modelo no detectó:

```text
ningún enfermo
```

***

## Consigna

¿Considerarías útil este modelo?

***

# Ejercicio 6 – Precision vs Recall

Supongamos dos hospitales.

***

## Modelo A

```text
Precision = 95%
Recall = 50%
```

***

## Modelo B

```text
Precision = 70%
Recall = 98%
```

***

## Explicación

Modelo A:

```text
Pocas falsas alarmas
```

Modelo B:

```text
Detecta casi todos los enfermos
```

***

## Consigna

¿Cuál elegirías para un sistema de detección temprana de cáncer?

Justifica tu respuesta.

***

# Ejercicio 7 – F1 Score

El F1 Score intenta equilibrar:

```text
Precision
Recall
```

***

## Código

```python
from sklearn.metrics import f1_score

y_real = [
    1,1,1,1,1,
    1,1,1,1,1
]

y_pred = [
    1,0,0,0,0,
    0,0,0,0,0
]

print(
    f1_score(
        y_real,
        y_pred
    )
)
```

***

## Explicación

Aunque la Precision puede parecer buena:

```text
Recall es muy baja
```

El F1 lo penaliza.

***

## Consigna

¿Por qué no basta con mirar únicamente Precision?

***

# Ejercicio 8 – Probabilidades y umbral

Muchos modelos generan probabilidades.

## Datos

```python
probabilidades = [
    0.95,
    0.90,
    0.85,
    0.70,
    0.60,
    0.40,
    0.30
]
```

***

## Regla habitual

```python
si probabilidad >= 0.5

    predice enfermo

si no

    predice sano
```

***

## Consigna

¿Qué ocurre si cambiamos el umbral a:

```text
0.8
```

?

¿Cuántos pacientes dejan de ser considerados enfermos?

***

# Ejercicio 9 – Curva ROC

La curva ROC estudia cómo cambia el modelo al mover el umbral.

***

## Umbral 0.9

```text
Muy estricto
```

***

## Umbral 0.5

```text
Intermedio
```

***

## Umbral 0.2

```text
Muy permisivo
```

***

## Código

```python
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

y_real = [
    1,1,1,1,0,0,0,0
]

probabilidades = [
    0.95,
    0.90,
    0.75,
    0.60,
    0.55,
    0.40,
    0.20,
    0.10
]

fpr, tpr, thresholds = roc_curve(
    y_real,
    probabilidades
)

plt.plot(
    fpr,
    tpr,
    marker="o"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "Curva ROC"
)

plt.show()
```

***

## Consigna

¿Qué representa cada punto de la curva?

***

# Ejercicio 10 – ROC AUC

ROC AUC resume la curva ROC en un único número.

***

## Código

```python
from sklearn.metrics import roc_auc_score

auc = roc_auc_score(
    y_real,
    probabilidades
)

print(auc)
```

***

## Interpretación

```text
1.00 = perfecto

0.90 = excelente

0.80 = bueno

0.70 = aceptable

0.50 = equivalente al azar
```

***

## Consigna

¿Por qué un ROC AUC de 0.50 es preocupante?

***

# Ejercicio 11 – Macro Average

Supongamos:

```text
Clase A = 1000 ejemplos
Clase B = 100 ejemplos
Clase C = 10 ejemplos
```

***

## Resultados

| Clase | F1   |
| ----- | ---- |
| A     | 0.90 |
| B     | 0.80 |
| C     | 0.20 |

***

## Macro Average

```text
(0.90 + 0.80 + 0.20) / 3
```

Resultado:

```text
0.63
```

***

## Explicación

Todas las clases tienen el mismo peso.

***

## Consigna

¿Por qué la clase C influye tanto en el resultado?

***

# Ejercicio 12 – Weighted Average

Usamos exactamente los mismos datos.

***

## Fórmula

```text
Cada clase pesa según
su cantidad de ejemplos
```

***

## Resultado aproximado

```text
0.89
```

***

## Explicación

La clase A domina porque tiene muchos ejemplos.

***

## Consigna

¿Cuándo sería más útil Macro Average que Weighted Average?

***

# Ejercicio 13 – Comparando clasificadores ficticios

## Tabla

| Modelo | Accuracy | Precision | Recall | F1  |
| ------ | -------- | --------- | ------ | --- |
| A      | 95%      | 96%       | 40%    | 57% |
| B      | 92%      | 88%       | 85%    | 86% |
| C      | 89%      | 75%       | 98%    | 85% |

***

## Consigna

Si el problema fuera:

### Diagnóstico médico

¿Cuál elegirías?

***

### Filtro de spam

¿Cuál elegirías?

***

### Detección de fraude

¿Cuál elegirías?

***

# Cierre del laboratorio

Con este laboratorio aprendiste:

* Qué es Accuracy.
* Qué muestra una matriz de confusión.
* Qué mide Precision.
* Qué mide Recall.
* Qué representa F1 Score.
* Cómo funciona ROC.
* Qué significa ROC AUC.
* Diferencia entre Macro Average y Weighted Average.
* Por qué Accuracy puede ser engañosa.
* Por qué la mejor métrica depende del problema.

## Idea clave

> No existe una métrica universalmente mejor. La métrica correcta depende del coste real de equivocarse en cada problema de negocio.
