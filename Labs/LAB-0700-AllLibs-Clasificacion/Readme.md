# Laboratorio: Clasificación simple con scikit-learn

## Objetivos

- Construir un dataset sintético con pandas para un problema de clasificación binaria.
- Entrenar un modelo de clasificación con scikit-learn.
- Evaluar el modelo con una matriz de confusión.
- Interpretar los resultados y sacar conclusiones.

## Dataset

Simulamos un caso simple: predecir si un estudiante aprueba un examen (`aprueba`) en función de `horas_estudio` y `asistencia` (%).

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report

np.random.seed(42)

n = 200
horas_estudio = np.random.uniform(0, 10, n)
asistencia = np.random.uniform(50, 100, n)

# Regla base con algo de ruido para que no sea perfectamente separable
score = (horas_estudio * 6) + (asistencia * 0.5) + np.random.normal(0, 10, n)
aprueba = (score > np.percentile(score, 50)).astype(int)

df = pd.DataFrame({
    "horas_estudio": horas_estudio,
    "asistencia": asistencia,
    "aprueba": aprueba
})

df.head()
```

## Entrenamiento

```python
X = df[["horas_estudio", "asistencia"]]
y = df["aprueba"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

modelo = LogisticRegression()
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)
```

## Matriz de confusión

```python
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No aprueba", "Aprueba"],
            yticklabels=["No aprueba", "Aprueba"])
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de confusión")
plt.show()
```

## Métricas

```python
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

## Consignas (responder por escrito)

1. ¿Cuántos falsos positivos y falsos negativos tuvo el modelo? ¿Qué significan en este contexto?
   
   **Respuesta:** Se leen directamente de la matriz de confusión: la celda (fila "No aprueba", columna "Aprueba") es la cantidad de falsos positivos — estudiantes que el modelo predijo que aprobaban pero en realidad no. La celda (fila "Aprueba", columna "No aprueba") son los falsos negativos — estudiantes que aprobaban pero el modelo predijo que no. Con `random_state=42` y este dataset, ambos valores suelen ser bajos (entre 3 y 8 sobre 60 casos de test), reflejando que la separación entre clases es buena pero no perfecta por el ruido agregado a `score`.

---
2. ¿El accuracy obtenido te parece suficiente para confiar en el modelo? Justificá.
   
   **Respuesta:** El accuracy solo (típicamente entre 0.80 y 0.90 en este dataset) no alcanza para confiar ciegamente en el modelo. Como las clases están balanceadas (50/50 por construcción), el accuracy es una medida razonable acá, pero igual conviene mirar precision/recall por clase: un accuracy alto puede esconder que el modelo falla sistemáticamente en un tipo de error (por ejemplo, muchos falsos negativos aunque pocos falsos positivos). La confianza en el modelo depende del costo de cada error, no solo del promedio.

---
3. Si tuvieras que priorizar reducir los falsos negativos (estudiantes que en realidad aprueban pero el modelo predice que no), ¿qué cambiarías del modelo o del umbral de decisión?
   
   **Respuesta:** Bajar el umbral de decisión de 0.5 a un valor menor (por ejemplo 0.35) usando `modelo.predict_proba(X_test)[:, 1] > umbral` en lugar de `predict()`. Esto hace que el modelo clasifique como "aprueba" con menos evidencia, reduciendo falsos negativos a costa de aumentar falsos positivos. Es el trade-off clásico de precision vs recall, y la elección del umbral depende de qué error es más costoso en el contexto real (por ejemplo, si el costo de decirle a alguien que no va a aprobar cuando sí iba a aprobar es alto).

---
4. ¿Qué pasaría con la matriz de confusión si `asistencia` no tuviera relación real con `aprueba`? Probá poniendo `asistencia` como ruido puro (`np.random.uniform(50, 100, n)` sin usarla en `score`) y comparar resultados.
    
   **Respuesta:** Al sacar `asistencia` del cálculo de `score`, esa variable deja de tener poder predictivo real y el modelo pasa a depender casi exclusivamente de `horas_estudio`. La matriz de confusión debería mostrar más errores (más falsos positivos y negativos) porque el modelo pierde una fuente de información válida, aunque siga "usando" `asistencia` como si aportara algo (los coeficientes de la regresión logística lo reflejan). Esto ilustra un punto importante: el modelo no distingue entre una variable útil y una que es ruido puro si no se lo mostramos explícitamente comparando ambos escenarios.

## Conclusiones esperadas

- La matriz de confusión permite ver no solo el accuracy global, sino el tipo de error que comete el modelo (falsos positivos vs falsos negativos), algo clave cuando el costo de cada error es distinto.
- Con datos sintéticos y ruido moderado, un modelo simple como regresión logística ya separa razonablemente bien las clases, pero rara vez llega a accuracy perfecto — y eso es esperable, no un error.

## Código completo

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report

np.random.seed(42)

# --- Dataset sintético ---
n = 200
horas_estudio = np.random.uniform(0, 10, n)
asistencia = np.random.uniform(50, 100, n)

score = (horas_estudio * 6) + (asistencia * 0.5) + np.random.normal(0, 10, n)
aprueba = (score > np.percentile(score, 50)).astype(int)

df = pd.DataFrame({
    "horas_estudio": horas_estudio,
    "asistencia": asistencia,
    "aprueba": aprueba
})

print(df.head())

# --- Entrenamiento ---
X = df[["horas_estudio", "asistencia"]]
y = df["aprueba"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

modelo = LogisticRegression()
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

# --- Matriz de confusión ---
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No aprueba", "Aprueba"],
            yticklabels=["No aprueba", "Aprueba"])
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de confusión")
plt.show()

# --- Métricas ---
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```
