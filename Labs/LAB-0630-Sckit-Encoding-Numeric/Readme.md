# Lab 0630 — Escalado de Variables Numéricas

## Objetivo

Aprender a escalar variables numéricas antes de entrenar modelos de Machine Learning.

Veremos los dos métodos más utilizados:

* **StandardScaler**
* **MinMaxScaler**

Y entenderemos en qué algoritmos son necesarios.

---

## ¿Por qué escalar?

Muchos algoritmos utilizan distancias o realizan optimización mediante descenso del gradiente.

Si una variable tiene valores mucho más grandes que otra, dominará el entrenamiento.

Por ejemplo:

| Edad | Sueldo |
| ---: | -----: |
|   25 |    800 |
|   40 |   1500 |
|   35 |   2500 |
|   60 |   6000 |
|   50 |  12000 |

Aquí **Sueldo** tiene valores mucho mayores que **Edad**.

Para un algoritmo como:

* Regresión Lineal
* Regresión Logística
* SVM
* KNN
* K-Means
* Redes Neuronales

esto puede perjudicar el entrenamiento.

---

## Dataset

```python
import pandas as pd

df = pd.DataFrame({
    "edad":[25,40,35,60,50],
    "sueldo":[800,1500,2500,6000,12000]
})

df
```

---

# Parte 1 — StandardScaler

Transforma cada variable para que tenga:

* media = 0
* desvío estándar = 1

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df_std = pd.DataFrame(
    scaler.fit_transform(df),
    columns=df.columns
)

df_std
```

### Observar

* Los valores ya no están en pesos ni años.
* Ambas columnas quedaron aproximadamente en la misma escala.

**Pregunta**

¿Cuál de las dos variables tiene ahora más "peso" para el algoritmo?

Respuesta esperada:

> Ninguna. Las dos quedaron comparables.

---

## ¿Cuándo se usa?

Muy recomendado para:

* Regresión Lineal
* Regresión Logística
* SVM
* KNN
* K-Means
* PCA
* Redes neuronales

---

# Parte 2 — MinMaxScaler

Ahora llevaremos todos los valores al rango **0-1**.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df_minmax = pd.DataFrame(
    scaler.fit_transform(df),
    columns=df.columns
)

df_minmax
```

### Observar

El valor mínimo de cada columna ahora vale **0**.

El máximo vale **1**.

Todos los demás quedaron entre ambos extremos.

---

## ¿Cuándo se usa?

Frecuente en:

* Redes neuronales
* Algoritmos que esperan entradas acotadas
* Algunos modelos basados en distancia

---

# Parte 3 — Comparación

```python
print("Original")
print(df)

print("\nStandardScaler")
print(df_std)

print("\nMinMaxScaler")
print(df_minmax)
```

---

## Comparación

| Método         | Resultado          | Rango               |
| -------------- | ------------------ | ------------------- |
| Original       | Sin cambios        | Variable            |
| StandardScaler | Media≈0 Desvío≈1   | No tiene límites    |
| MinMaxScaler   | Escala entre 0 y 1 | Siempre entre 0 y 1 |

---

## ¿Cuál elegir?

**StandardScaler**

✔ Es el más utilizado.

✔ Funciona muy bien para la mayoría de los algoritmos lineales.

✔ Es el escalado por defecto en muchos proyectos.

---

**MinMaxScaler**

✔ Cuando se necesita que todos los valores estén entre 0 y 1.

✔ Muy utilizado en redes neuronales.

---

# Ejercicio

Agregar una nueva columna llamada **antigüedad**.

```python
[1,5,8,20,12]
```

Aplicar ambos escaladores y comparar cómo cambia la nueva variable.

---

## 💡 Una mejora que haría respecto a tu laboratorio anterior

En el de *Encoding* el alumno descubre el problema de `get_dummies()` frente a `OneHotEncoder`. Acá haría algo similar: **mostrar que hay algoritmos a los que el escalado no les importa**.

Al final agregaría esta sección:

## ¿Todos los algoritmos necesitan escalado?

| Algoritmo                     | ¿Escalar?           |
| ----------------------------- | ------------------- |
| Regresión Lineal              | ✅ Sí                |
| Regresión Logística           | ✅ Sí                |
| SVM                           | ✅ Sí                |
| KNN                           | ✅ Sí                |
| K-Means                       | ✅ Sí                |
| Redes neuronales              | ✅ Sí                |
| Árbol de decisión             | ❌ No                |
| Random Forest                 | ❌ No                |
| XGBoost / LightGBM / CatBoost | ❌ No (generalmente) |

Ese cuadro conecta perfectamente con la siguiente clase de algoritmos: los alumnos entienden que **el preprocesamiento depende del modelo**, no es un paso universal. Es un mensaje muy valioso para quienes recién empiezan con Machine Learning.
