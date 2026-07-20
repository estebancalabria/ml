# LAB-750 – Regresión - ML: California Housing con Regresión Lineal y Random Forest

## Objetivo

Entrenar y comparar dos modelos de regresión (LinearRegression y RandomForestRegressor) sobre el dataset California Housing, usar la matriz de correlación para entender qué variables pesan más, y evaluar con las métricas de regresión que ya vimos (MAE, MSE, RMSE, R²).

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

- `fetch_california_housing(as_frame=True)`: descarga (o usa la copia en caché local) el dataset y, con `as_frame=True`, nos lo devuelve directamente como DataFrame de pandas en vez de arrays sueltos.
- `datos.frame`: el DataFrame completo, con las 8 features **y** el target `MedHouseVal` ya en la misma tabla.

> [!NOTE]
> A diferencia de Iris o Titanic, acá **no hay nulos y no hace falta encoding**: las 9 columnas ya vienen numéricas y completas. Vamos directo a explorar la relación entre variables y a entrenar.

### Consigna

Mirando la tabla, ¿cuál de las 8 features te parece, a ojo, la que más debería influir en el precio de una vivienda?

---

## Ejercicio 1 – Matriz de correlación

Antes de entrenar cualquier modelo, miramos cómo se relaciona cada feature con el target y entre sí (colinealidad).

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

### Explicación línea por línea

- `df.corr()`: matriz de correlación de Pearson entre las 9 columnas numéricas.
- `matriz_correlacion["MedHouseVal"]`: nos quedamos solo con la columna que nos importa primero — cuánto se relaciona cada feature con el precio.
- El resto del código es el mismo heatmap que ya usamos en Titanic y en la Clase 3: `imshow` + `colorbar` + texto en cada celda.

### Lectura de la matriz

- **`MedInc` (ingreso medio) = 0.69**: es, por lejos, la variable más correlacionada con el precio. Tiene sentido: donde vive gente con más ingreso, las casas valen más.
- El resto de las features (`HouseAge`, `AveRooms`, `Population`, `AveOccup`, `Latitude`, `Longitude`) tienen correlaciones lineales bajas con el target (todas entre -0.14 y 0.15). Eso **no** significa que no aporten nada — puede haber relaciones no lineales (por ejemplo, la ubicación geográfica es clave para el precio real, pero su relación con `Latitude`/`Longitude` por separado no es lineal) — pero sí nos dice que, si tuviéramos que apostar a una sola variable para un modelo simple, es `MedInc`.
- No se ve colinealidad fuerte entre features (ningún par fuera de la diagonal se acerca a ±1), así que no hace falta descartar ninguna columna por ese motivo.

### Consigna

¿Por qué te parece que `Latitude` y `Longitude` solas tienen tan poca correlación lineal con el precio, si sabemos que la ubicación en California (cerca de la costa, cerca de San Francisco, etc.) es clave para el valor de una propiedad?

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
- `test_size=0.2`: dejamos 20% para test (con 20.640 filas totales, alcanza para tener un test set grande y confiable).
- `random_state=42`: reproducibilidad, como en los labs anteriores.

### Consigna

En Titanic usamos `test_size=0.3` y acá `0.2`. ¿Por qué con más datos totales (20.640 vs 418) podemos "darnos el lujo" de reservar un porcentaje menor para test?

---

## Ejercicio 3 – Entrenar Regresión Lineal (baseline)

Como vimos en la Clase 2, siempre conviene arrancar con un modelo baseline simple antes de ir a algo más complejo.

### Código

```python
from sklearn.linear_model import LinearRegression

modelo_lr = LinearRegression()
modelo_lr.fit(X_train, y_train)

pred_lr = modelo_lr.predict(X_test)

for columna, coeficiente in zip(X.columns, modelo_lr.coef_):
    print(f"{columna}: {coeficiente:.4f}")
print("Intercepto:", modelo_lr.intercept_)
```

### Salida

```
MedInc: 0.4487
HouseAge: 0.0097
AveRooms: -0.1233
AveBedrms: 0.7831
Population: -0.0000
AveOccup: -0.0035
Latitude: -0.4198
Longitude: -0.4337
Intercepto: -37.0233
```

### Explicación línea por línea

- `LinearRegression().fit(X_train, y_train)`: entrena, buscando el coeficiente que multiplica a cada feature (más el intercepto) que mejor ajusta el precio.
- Cada coeficiente indica cuánto sube o baja la predicción de `MedHouseVal` por cada unidad que aumenta esa feature, **manteniendo las demás constantes**.
- Fijate que `MedInc` tiene coeficiente positivo (0.45): a más ingreso, más precio, como esperábamos por la correlación del Ejercicio 1.
- `Latitude` y `Longitude` tienen coeficientes grandes y negativos. No hay que leerlos como "ir más al norte baja el precio" de forma simple — en un modelo lineal, la ubicación geográfica se está usando como una aproximación cruda de "cerca o lejos de la costa/las ciudades caras", y por eso pesa tanto aunque la correlación individual del Ejercicio 1 fuera baja.

### Consigna

`Population` tiene un coeficiente prácticamente 0 (-0.0000). ¿Te parece que esa variable no aporta nada al modelo, o puede estar relacionada con la escala en la que está medida (cientos de miles de habitantes) comparada con las demás?

---

## Ejercicio 4 – Entrenar Random Forest

### Código

```python
from sklearn.ensemble import RandomForestRegressor

modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
modelo_rf.fit(X_train, y_train)

pred_rf = modelo_rf.predict(X_test)
```

### Explicación línea por línea

- `RandomForestRegressor`: la versión de Random Forest para regresión. En vez de votar por una clase (como en Titanic), cada árbol predice un número y el bosque promedia esas 100 predicciones.
- `n_estimators=100`: cantidad de árboles del bosque.
- `n_jobs=-1`: usa todos los núcleos disponibles del procesador para entrenar más rápido (no afecta el resultado, solo la velocidad).

### Consigna

A diferencia de la regresión lineal, Random Forest no tiene "coeficientes" que uno pueda leer directamente. ¿Qué atributo del modelo (pensá en lo que vimos con `DecisionTreeClassifier` en la Clase 4) te imaginás que existe para saber qué features usó más el bosque?

---

## Ejercicio 5 – Comparar métricas

### Código

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

for nombre, predicciones in [("LinearRegression", pred_lr), ("RandomForestRegressor", pred_rf)]:
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

--- RandomForestRegressor ---
MAE : 0.3299
MSE : 0.2569
RMSE: 0.5068
R2  : 0.8040
```

### Lectura de las métricas

- **R²**: la regresión lineal explica el 44.6% de la variación del precio; Random Forest, el 80.4%. Recordando la Clase 4: R² > 0.8 indica que las variables están bien relacionadas con el target — Random Forest cruza justo ese umbral, la lineal se queda bastante por debajo.
- **MAE**: en promedio, la regresión lineal se equivoca por 0.55 (unos $55.000), Random Forest por 0.33 (unos $33.000).
- Esto tiene sentido con lo que vimos en el Ejercicio 1: como la relación entre las features (sobre todo `Latitude`/`Longitude`) y el precio **no es puramente lineal**, un modelo que sí puede capturar relaciones no lineales (Random Forest) le saca ventaja considerable a la regresión lineal acá.

### Consigna

Ninguno de los dos modelos llega a R² = 1.0 (a diferencia de lo que nos pasó -sospechosamente- con Random Forest en el lab de Titanic). ¿Te parece razonable o preocupante que un modelo de precios de viviendas reales no prediga perfecto? ¿Por qué habrá tanta variabilidad que ninguna de estas 8 variables explica?

---

## Ejercicio 6 – Gráfico de predicción vs. valor real

Un gráfico clásico en regresión: si el modelo fuera perfecto, todos los puntos caerían sobre la diagonal.

### Código

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 6))

plt.scatter(y_test, pred_lr, alpha=0.3, label="LinearRegression", s=10)
plt.scatter(y_test, pred_rf, alpha=0.3, label="RandomForestRegressor", s=10)

# La diagonal ideal: predicción == valor real
limite = [y_test.min(), y_test.max()]
plt.plot(limite, limite, color="black", linestyle="--", linewidth=1, label="Predicción perfecta")

plt.xlabel("Valor real (MedHouseVal)")
plt.ylabel("Predicción")
plt.title("Predicción vs Valor real")
plt.legend()
plt.show()
```

### Explicación línea por línea

- `plt.scatter(y_test, pred_lr, alpha=0.3, ...)`: en el eje X ponemos el valor real, en el eje Y la predicción. `alpha=0.3` hace los puntos semi-transparentes, porque con 4.128 puntos se superponen mucho.
- La línea punteada negra marca la "predicción perfecta" (`y_pred == y_real`). Cuanto más pegados estén los puntos de un modelo a esa línea, mejor predice.
- Vas a ver que la nube naranja (Random Forest) se pega más a la diagonal que la azul (LinearRegression), sobre todo en los valores altos.

### Consigna

Vas a notar que ambos modelos tienden a "aplastarse" cerca de MedHouseVal = 5.0 (el valor real tiene un techo ahí, producto de cómo se recolectaron originalmente los datos). ¿Cómo creés que afecta ese techo artificial a la métrica R² calculada sobre esos casos?

---

## Cierre del laboratorio

Con este laboratorio cerramos el circuito de regresión: matriz de correlación para entender qué variables pesan antes de entrenar, un modelo baseline lineal, un modelo más complejo (Random Forest) para comparar, y las cuatro métricas de regresión (MAE, MSE, RMSE, R²) aplicadas a un caso con datos reales y ruidosos — a diferencia de Titanic, acá no hay ningún atajo perfecto escondido: la dificultad de predecir precios de vivienda es genuina.
