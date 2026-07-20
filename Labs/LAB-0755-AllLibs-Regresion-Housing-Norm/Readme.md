# LAB-1000 – Regresión - ML: California Housing con Normalización, KNN y SVM

## Objetivo

Entrenar y comparar cuatro modelos de regresión (LinearRegression, KNN, SVM y Random Forest) sobre el dataset California Housing. Vamos a ver por qué normalizar las variables es obligatorio para algunos modelos (KNN, SVM) e indiferente para otros (Random Forest), y a evaluar todo con las métricas de regresión ya conocidas (MAE, MSE, RMSE, R²).

## Requisitos previos

```
pip install scikit-learn matplotlib pandas
```

## El dataset

`fetch_california_housing` viene incluido en scikit-learn, no hace falta descargar nada aparte. Son 20.640 bloques censales de California, con 8 features numéricas y un target continuo: el valor medio de las viviendas de ese bloque (en cientos de miles de dólares).

```python
from sklearn.datasets import fetch_california_housing
import pandas as pd

datos = fetch_california_housing(as_frame=True)
df = datos.frame

print(df.shape)
print(df.columns.tolist())
df.head()
```

### Salida

```
(20640, 9)
['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude', 'MedHouseVal']
```

| Columna | Significado |
|---|---|
| MedInc | Ingreso medio del bloque (en decenas de miles de USD) |
| HouseAge | Antigüedad media de las viviendas del bloque |
| AveRooms | Promedio de habitaciones por hogar |
| AveBedrms | Promedio de dormitorios por hogar |
| Population | Población del bloque |
| AveOccup | Promedio de ocupantes por hogar |
| Latitude / Longitude | Ubicación geográfica |
| **MedHouseVal** | **Target**: valor medio de la vivienda (en cientos de miles de USD, ej: 2.07 = $207.000) |

### Explicación línea por línea

- `fetch_california_housing(as_frame=True)`: descarga (o usa la copia en caché local) el dataset y nos lo devuelve como DataFrame de pandas.
- `datos.frame`: el DataFrame completo, con las 8 features **y** el target `MedHouseVal` en la misma tabla.

> [!NOTE]
> No hay nulos y no hay columnas de texto, así que no hace falta encoding acá (a diferencia de Titanic). Pero fijate en la tabla algo importante para más adelante: `Population` está en cientos/miles, `MedInc` está en decenas de unidades, y `Latitude`/`Longitude` rondan los 30-120. **Las escalas son muy distintas entre columnas.** Eso va a ser el eje de este laboratorio.

### Consigna

Mirando las escalas tan distintas de `Population` (miles) contra `AveBedrms` (números como 1.0, 1.1, 1.2), ¿qué problema te imaginás que le puede generar eso a un modelo que mide "distancia" entre puntos, como KNN?

---

## Ejercicio 1 – Matriz de correlación

### Código

```python
import matplotlib.pyplot as plt

matriz_correlacion = df.corr()

print(matriz_correlacion["MedHouseVal"].round(2))

plt.figure(figsize=(8, 7))
plt.imshow(matriz_correlacion, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlación")

plt.xticks(range(len(matriz_correlacion.columns)), matriz_correlacion.columns, rotation=45, ha="right")
plt.yticks(range(len(matriz_correlacion.columns)), matriz_correlacion.columns)

for i in range(len(matriz_correlacion)):
    for j in range(len(matriz_correlacion)):
        plt.text(j, i, f"{matriz_correlacion.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)

plt.title("Matriz de Correlación - California Housing")
plt.tight_layout()
plt.show()
```

### Salida

```
MedInc         0.69
HouseAge       0.11
AveRooms       0.15
AveBedrms     -0.05
Population    -0.02
AveOccup      -0.02
Latitude      -0.14
Longitude     -0.05
MedHouseVal    1.00
Name: MedHouseVal, dtype: float64
```

### Lectura de la matriz

- **`MedInc` (ingreso medio) = 0.69**: la variable más correlacionada con el precio, por lejos.
- El resto tiene correlaciones lineales bajas con el target. No implica que no aporten — puede haber relaciones no lineales que la correlación de Pearson no capta — pero `MedInc` es la más fuerte a simple vista.
- No hay colinealidad fuerte entre features (ningún par fuera de la diagonal se acerca a ±1).

### Consigna

¿Por qué te parece que `Latitude` y `Longitude` solas tienen tan poca correlación lineal con el precio, si sabemos que la ubicación en California es clave para el valor de una propiedad?

---

## Ejercicio 2 – Train/Test Split

### Código

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train:", X_train.shape)
print("Test:", X_test.shape)
```

### Salida

```
Train: (16512, 8)
Test: (4128, 8)
```

### Explicación línea por línea

- `X`: las 8 features. `y`: el target `MedHouseVal`.
- `test_size=0.2`: con 20.640 filas totales, un 20% para test ya nos da un conjunto grande y confiable.
- `random_state=42`: reproducibilidad.

---

## Ejercicio 3 – El problema de no normalizar: KNN "a lo bruto"

Antes de normalizar nada, probemos qué pasa si entrenamos KNN directamente sobre los datos crudos, con las escalas tan distintas que vimos en la introducción.

### Código

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score

modelo_knn_sin_escalar = KNeighborsRegressor(n_neighbors=5)
modelo_knn_sin_escalar.fit(X_train, y_train)

pred_knn_sin_escalar = modelo_knn_sin_escalar.predict(X_test)

print("MAE:", mean_absolute_error(y_test, pred_knn_sin_escalar))
print("R2 :", r2_score(y_test, pred_knn_sin_escalar))
```

### Salida

```
MAE: 0.8126
R2 : 0.1466
```

### Explicación

Un R² de 0.15 es malísimo (recordemos: 0 significa "no explica nada", y la regresión lineal sin escalar del lab anterior ya daba 0.45). El motivo es el que sospechaste en la introducción: KNN calcula la distancia euclidiana entre puntos para encontrar los "vecinos más cercanos". Si `Population` va de 3 a 35.000 y `AveBedrms` va de 0.8 a 5, la distancia queda **totalmente dominada por `Population`** — las demás columnas casi no pesan en la cuenta, aunque `MedInc` sea mucho más relevante para el precio. KNN, tal como está acá, básicamente está agrupando bloques censales por población y furreando el resto.

### Consigna

¿Te parece que un modelo basado en árboles (como Random Forest, que vimos en el lab anterior) tendría el mismo problema con estas escalas tan distintas? Pensalo en términos de cómo decide un árbol dónde "cortar" una variable.

---

## Ejercicio 4 – Normalizar con StandardScaler

### Código

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_escalado = scaler.fit_transform(X_train)
X_test_escalado = scaler.transform(X_test)

print("Medias antes de escalar:")
print(X_train.mean().round(2))

print()
print("Medias después de escalar (deberían ser ~0):")
print(pd.DataFrame(X_train_escalado, columns=X.columns).mean().round(2))

print()
print("Desvíos después de escalar (deberían ser ~1):")
print(pd.DataFrame(X_train_escalado, columns=X.columns).std().round(2))
```

### Salida

```
Medias antes de escalar:
MedInc           3.88
HouseAge        28.61
AveRooms         5.44
AveBedrms        1.10
Population    1426.45
AveOccup         3.10
Latitude        35.64
Longitude     -119.58
dtype: float64

Medias después de escalar (deberían ser ~0):
MedInc       -0.0
HouseAge     -0.0
AveRooms      0.0
AveBedrms    -0.0
Population    0.0
AveOccup      0.0
Latitude      0.0
Longitude     0.0
dtype: float64

Desvíos después de escalar (deberían ser ~1):
MedInc        1.0
HouseAge      1.0
AveRooms      1.0
AveBedrms     1.0
Population    1.0
AveOccup      1.0
Latitude      1.0
Longitude     1.0
dtype: float64
```

### Explicación línea por línea

- `StandardScaler()`: transforma cada columna para que quede con media 0 y desvío estándar 1. La fórmula es `(x - media) / desvío`.
- `scaler.fit_transform(X_train)`: **calcula** la media y el desvío usando *solo* los datos de entrenamiento, y transforma `X_train` con esos valores.
- `scaler.transform(X_test)`: transforma `X_test` reusando la media y el desvío ya calculados en el paso anterior — **sin volver a calcularlos con los datos de test**.

> [!NOTE]
> Este es un detalle que se presta a errores: nunca hay que hacer `fit_transform` sobre `X_test`. Si el scaler "aprendiera" la media y el desvío de los datos de test, estaríamos filtrando información de test hacia el entrenamiento (un tipo de data leakage), aunque sea algo tan sutil como una media.

- Después de transformar, las 8 columnas quedan en la misma escala: media 0, desvío 1. Ahora sí son comparables entre sí para calcular distancias.

### Consigna

¿Por qué el scaler se entrena (`fit`) solo con `X_train` y no con todo el dataset antes de hacer el split?

---

## Ejercicio 5 – Entrenar los cuatro modelos (ya escalados)

Ahora sí, entrenamos LinearRegression, KNN, SVM y Random Forest sobre los datos normalizados, para comparar en igualdad de condiciones.

### Código

```python
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

modelos = {
    "LinearRegression": LinearRegression(),
    "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=5),
    "SVR (SVM para regresión)": SVR(kernel="rbf"),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
}

resultados = {}

for nombre, modelo in modelos.items():
    modelo.fit(X_train_escalado, y_train)
    predicciones = modelo.predict(X_test_escalado)
    resultados[nombre] = predicciones

    mae = mean_absolute_error(y_test, predicciones)
    mse = mean_squared_error(y_test, predicciones)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predicciones)

    print(f"--- {nombre} ---")
    print(f"MAE : {mae:.4f}")
    print(f"MSE : {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2  : {r2:.4f}")
    print()
```

### Salida

```
--- LinearRegression ---
MAE : 0.5542
MSE : 0.7254
RMSE: 0.8517
R2  : 0.4464

--- KNeighborsRegressor ---
MAE : 0.4517
MSE : 0.4443
RMSE: 0.6665
R2  : 0.6610

--- SVR (SVM para regresión) ---
MAE : 0.4104
MSE : 0.3757
RMSE: 0.6130
R2  : 0.7133

--- RandomForestRegressor ---
MAE : 0.3298
MSE : 0.2567
RMSE: 0.5066
R2  : 0.8041
```

### Explicación línea por línea

- `SVR(kernel="rbf")`: es la versión de SVM (Support Vector Machine) para regresión. `SVC` es para clasificación, `SVR` para predecir un número continuo. El kernel `"rbf"` (Radial Basis Function) mide qué tan "cerca" está un punto de otro — por eso, igual que KNN, **es sensible a la escala** de las features.
- El resto del bucle es igual al lab anterior: entrena, predice, calcula las 4 métricas.

### Comparación

| Modelo | MAE | R² |
|---|---|---|
| LinearRegression | 0.554 | 0.446 |
| KNeighborsRegressor | 0.452 | 0.661 |
| SVR | 0.410 | 0.713 |
| RandomForestRegressor | 0.330 | **0.804** |

Comparado con el Ejercicio 3 (KNN sin escalar, R² = 0.15), el mismo KNN ya escalado pasa a R² = 0.66 — la normalización fue la diferencia entre un modelo inútil y uno razonable. SVM también queda muy por encima de la regresión lineal. Random Forest sigue siendo el mejor de los cuatro, y no necesitaba el escalado para nada (podés confirmarlo corriendo este mismo ejercicio con `X_train`/`X_test` sin escalar y comparando: te va a dar prácticamente el mismo resultado).

### Consigna

KNN y SVR quedaron mejor que LinearRegression pero peor que Random Forest. ¿Qué tienen en común KNN y SVR (con kernel RBF) que los hace más flexibles que una recta, pero aun así no tan buenos como un bosque de árboles acá?

---

## Ejercicio 6 – Coeficientes de la regresión lineal, ahora comparables

Con las variables ya en la misma escala, los coeficientes de LinearRegression se pueden comparar directamente entre sí (algo que no podíamos hacer bien con los datos crudos, donde `Population` daba un coeficiente casi 0 solo por su escala).

### Código

```python
modelo_lr_escalado = LinearRegression()
modelo_lr_escalado.fit(X_train_escalado, y_train)

coeficientes = sorted(
    zip(X.columns, modelo_lr_escalado.coef_),
    key=lambda x: -abs(x[1])
)

for columna, coeficiente in coeficientes:
    print(f"{columna}: {coeficiente:.4f}")
```

### Salida

```
Latitude: -0.8969
Longitude: -0.8698
MedInc: 0.8544
AveBedrms: 0.3393
AveRooms: -0.2944
HouseAge: 0.1225
AveOccup: -0.0408
Population: -0.0023
```

### Explicación

Ahora que las 8 features están en la misma escala, el tamaño del coeficiente sí refleja qué tan importante es cada variable para el modelo. `Latitude`, `Longitude` y `MedInc` son las tres que más pesan. `Population` sigue siendo la que menos aporta — pero ahora lo sabemos porque su coeficiente escalado es chico (-0.0023), no porque su escala original (miles) lo hiciera parecer chico artificialmente.

### Consigna

Si entrenaras esta misma regresión lineal con los datos **sin** escalar y compararas los coeficientes, ¿cambiaría el orden de importancia real de las variables, o solo cómo se ve reflejado en el número del coeficiente?

---

## Cierre del laboratorio

La gran lección de este laboratorio: **la normalización no es un paso "prolijo" opcional, es indispensable para ciertos algoritmos**. Vimos en carne propia cómo KNN pasó de R² = 0.15 (inútil) a R² = 0.66 (razonable) con el único cambio de escalar las variables, y que SVM también depende de ese mismo paso. Random Forest, en cambio, no lo necesitó — porque los árboles deciden por umbrales en cada variable por separado, no por distancia entre puntos. La regla general: si el algoritmo mide distancia (KNN, SVM, K-Means, PCA) hay que normalizar sí o sí; si está basado en árboles (Decision Tree, Random Forest, Gradient Boosting) es indiferente.
