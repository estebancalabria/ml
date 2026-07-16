# Lab 0730 — K-Nearest Neighbors (KNN) y Escalado de Variables

## Objetivo

Entrenar un modelo **K-Nearest Neighbors (KNN)** para clasificar clientes y comprobar cómo el **escalado de variables numéricas** puede afectar el resultado del modelo.

Utilizaremos:

* Sin escalado
* `StandardScaler`

Al finalizar comprenderemos por qué KNN necesita que las variables estén en una escala similar.

---

# ¿Por qué escalar?

KNN clasifica un nuevo dato buscando los **K vecinos más cercanos**.

La cercanía se calcula mediante una distancia (por defecto, la **distancia euclídea**).

Si una variable tiene valores mucho mayores que otra, dominará el cálculo de la distancia.

Por ejemplo:

| Edad | Ingreso |
| ---: | ------: |
|   25 |  30.000 |
|   40 |  90.000 |

Aunque la diferencia de edad sea importante, el ingreso tiene una escala muchísimo mayor, por lo que influirá mucho más en el cálculo de la distancia.

Por eso, antes de utilizar KNN es recomendable escalar las variables.

---

# Dataset

Vamos a crear un pequeño dataset manualmente.

```python
import pandas as pd

df = pd.DataFrame({
    "edad": [22, 25, 28, 35, 40, 45, 50, 55],
    "ingreso": [30000, 35000, 45000, 60000, 75000, 90000, 120000, 150000],
    "compra": ["No", "No", "No", "Si", "Si", "Si", "Si", "Si"]
})

df
```

---

# Parte 1 — Preparar los datos

La variable objetivo es categórica ("Si" / "No"), por lo que la convertimos a valores numéricos.

```python
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder(categories=[["No", "Si"]])

df["compra_enc"] = encoder.fit_transform(df[["compra"]])

df
```

---

# Parte 2 — Entrenar KNN sin escalado

Seleccionamos las variables predictoras.

```python
X = df[["edad", "ingreso"]]
y = df["compra_enc"]
```

Entrenamos el modelo.

```python
from sklearn.neighbors import KNeighborsClassifier

modelo = KNeighborsClassifier(n_neighbors=3)

modelo.fit(X, y)
```

---

# Parte 3 — Realizar una predicción

Queremos clasificar un nuevo cliente.

```python
nuevo_cliente = pd.DataFrame({
    "edad": [30],
    "ingreso": [50000]
})

prediccion = modelo.predict(nuevo_cliente)

print(prediccion)
```

Convertimos nuevamente el resultado a texto.

```python
encoder.inverse_transform(prediccion.reshape(-1,1))
```

---

# Parte 4 — Escalar las variables

Ahora aplicaremos `StandardScaler`.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_scaled
```

Escalamos también el nuevo cliente utilizando **el mismo scaler**.

```python
nuevo_cliente_scaled = scaler.transform(nuevo_cliente)
```

> **Importante:** Nunca debemos hacer `fit()` sobre los datos nuevos. El escalador se ajusta únicamente con los datos de entrenamiento y luego se reutiliza con `transform()`.

---

# Parte 5 — Entrenar nuevamente el modelo

```python
modelo_scaled = KNeighborsClassifier(n_neighbors=3)

modelo_scaled.fit(X_scaled, y)
```

Realizamos nuevamente la predicción.

```python
prediccion_scaled = modelo_scaled.predict(nuevo_cliente_scaled)

encoder.inverse_transform(prediccion_scaled.reshape(-1,1))
```

---

# Parte 6 — Comparar los resultados

Comparar:

* Predicción sin escalado
* Predicción con `StandardScaler`

Responder:

* ¿Cambió la clasificación?
* ¿Por qué creen que ocurrió?

---

# ¿Qué cambió?

Antes del escalado:

| Variable | Rango aproximado |
| -------- | ---------------: |
| Edad     |          22 - 55 |
| Ingreso  | 30.000 - 150.000 |

La variable **Ingreso** tiene valores muchísimo mayores y domina el cálculo de la distancia.

Después del escalado:

| Variable | Escala    |
| -------- | --------- |
| Edad     | Media ≈ 0 |
| Ingreso  | Media ≈ 0 |

Ahora ambas variables tienen una importancia similar para el algoritmo.

---

# ¿Con qué algoritmos es recomendable escalar?

| Algoritmo                    | ¿Escalar? |
| ---------------------------- | --------- |
| K-Nearest Neighbors (KNN)    | ✅ Sí      |
| Regresión Lineal             | ✅ Sí      |
| Regresión Logística          | ✅ Sí      |
| Support Vector Machine (SVM) | ✅ Sí      |
| Redes Neuronales             | ✅ Sí      |
| PCA                          | ✅ Sí      |
| K-Means                      | ✅ Sí      |
| Árbol de Decisión            | ❌ No      |
| Random Forest                | ❌ No      |

---

# Cierre — Puntos para discutir en clase

1. ¿Por qué el ingreso influye mucho más que la edad cuando no escalamos los datos?
2. ¿Qué sucede si todas las variables tienen escalas similares?
3. ¿Por qué utilizamos el mismo `StandardScaler` para transformar el nuevo cliente?
4. ¿Qué ocurriría si entrenáramos el modelo con datos escalados pero luego predijéramos usando datos sin escalar?

---

# Ejercicio para los alumnos

1. Cambiar `StandardScaler` por `MinMaxScaler` y comparar las predicciones.
2. Agregar una nueva variable numérica llamada **antigüedad_laboral** y volver a entrenar el modelo.
3. Cambiar `n_neighbors` de **3** a **5** y analizar cómo cambia la clasificación.
4. Explicar con sus palabras por qué **KNN necesita variables escaladas**, mientras que un **Árbol de Decisión** puede funcionar correctamente sin realizar este preprocesamiento.
