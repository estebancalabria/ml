# LAB-0610 · Scikit-Learn: Regresión Lineal y Correlación

## Objetivo

Comprender, mediante dos conjuntos de datos hardcodeados (horas de estudio vs. nota obtenida), cómo se calcula una regresión lineal simple con `scikit-learn`, qué representan sus errores (residuos, MAE, MSE, RMSE) y qué significa el coeficiente **R²**. Se comparará un caso con correlación lineal fuerte contra un caso sin correlación, para que el R² y los errores muestren esa diferencia de forma concreta.

## En este laboratorio aprenderás...

- A entrenar un modelo de regresión lineal simple con `LinearRegression` de scikit-learn.
- Qué es un residuo (error) y cómo se calcula para cada observación.
- Qué miden MAE, MSE y RMSE, y en qué unidades se interpretan.
- Qué significa el R² (coeficiente de determinación) y por qué varía entre 0 y 1 (o incluso puede ser negativo).
- A distinguir, con evidencia numérica, cuándo una relación lineal existe y cuándo no.

---

## Paso 1 — Preparar el entorno

```bash
pip install scikit-learn numpy
```

Importamos las librerías necesarias:

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

---

## Paso 2 — Dataset A: horas de estudio y nota (CON correlación lineal)

Este dataset simula una situación esperable: a más horas de estudio, mejor nota. La relación no es perfecta (hay algo de ruido), pero es claramente lineal.

```python
horas_A = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
notas_A = np.array([3.2, 3.8, 4.5, 5.1, 5.9, 6.4, 7.0, 7.8, 8.3, 9.1])
```

**Idea clave:** `horas_A` es la variable independiente (X, predictora) y `notas_A` es la variable dependiente (y, lo que queremos predecir).

---

## Paso 3 — Dataset B: horas de estudio y nota (SIN correlación lineal)

Este segundo dataset representa un caso donde la nota no depende linealmente de las horas estudiadas (por ejemplo, otros factores dominan: ansiedad, calidad del estudio, suerte en el examen, etc.).

```python
horas_B = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
notas_B = np.array([6.5, 4.0, 7.8, 5.2, 6.9, 4.5, 7.1, 5.8, 6.0, 5.5])
```

Fijate que los valores de nota "saltan" sin un patrón claro respecto a las horas.

---

## Paso 4 — Entrenar el modelo con el Dataset A

```python
modelo_A = LinearRegression()
modelo_A.fit(horas_A, notas_A)

pred_A = modelo_A.predict(horas_A)

print("Pendiente (coef_):", modelo_A.coef_[0])
print("Ordenada al origen (intercept_):", modelo_A.intercept_)
```

**Qué significa esto:**
- `coef_` es cuánto sube la nota, en promedio, por cada hora adicional de estudio.
- `intercept_` es la nota estimada cuando las horas de estudio son 0 (el punto donde la recta cruza el eje Y).

La recta ajustada tiene la forma: `nota = coef_ * horas + intercept_`

---

## Paso 5 — Calcular los errores del Dataset A

Un **error** (o **residuo**) es la diferencia entre el valor real y el valor que predijo el modelo, para cada observación:

```python
residuos_A = notas_A - pred_A
print("Residuos individuales:", residuos_A)
```

Ahora las métricas agregadas:

```python
mae_A = mean_absolute_error(notas_A, pred_A)
mse_A = mean_squared_error(notas_A, pred_A)
rmse_A = np.sqrt(mse_A)
r2_A = r2_score(notas_A, pred_A)

print(f"MAE:  {mae_A:.3f}")
print(f"MSE:  {mse_A:.3f}")
print(f"RMSE: {rmse_A:.3f}")
print(f"R2:   {r2_A:.3f}")
```

**Qué significa cada métrica:**

- **MAE (Mean Absolute Error):** el promedio de cuánto se equivoca el modelo, en las mismas unidades que la nota. Si MAE = 0.3, en promedio el modelo se equivoca por 0.3 puntos de nota.
- **MSE (Mean Squared Error):** similar al MAE pero elevando cada error al cuadrado antes de promediar. Esto penaliza más fuerte los errores grandes. Está en unidades al cuadrado (notas²), por lo que es menos intuitivo de interpretar solo.
- **RMSE (Root Mean Squared Error):** la raíz cuadrada del MSE. Vuelve a las unidades originales (notas), y sigue penalizando más los errores grandes que el MAE.
- **R² (coeficiente de determinación):** indica qué proporción de la variabilidad de la nota es explicada por las horas de estudio. Va de 0 a 1 (más cercano a 1 = mejor ajuste lineal). Un R² de 0.95 significa que el 95% de la variación en la nota se explica por la recta de regresión.

---

## Paso 6 — Entrenar el modelo con el Dataset B

Repetimos exactamente el mismo procedimiento, pero con el dataset sin correlación:

```python
modelo_B = LinearRegression()
modelo_B.fit(horas_B, notas_B)

pred_B = modelo_B.predict(horas_B)

mae_B = mean_absolute_error(notas_B, pred_B)
mse_B = mean_squared_error(notas_B, pred_B)
rmse_B = np.sqrt(mse_B)
r2_B = r2_score(notas_B, pred_B)

print(f"MAE:  {mae_B:.3f}")
print(f"MSE:  {mse_B:.3f}")
print(f"RMSE: {rmse_B:.3f}")
print(f"R2:   {r2_B:.3f}")
```

---

## Paso 7 — Comparar ambos casos

```python
print("=== Dataset A (con correlación) ===")
print(f"R2: {r2_A:.3f} | MAE: {mae_A:.3f} | RMSE: {rmse_A:.3f}")

print("=== Dataset B (sin correlación) ===")
print(f"R2: {r2_B:.3f} | MAE: {mae_B:.3f} | RMSE: {rmse_B:.3f}")
```

**Qué deberías observar:**

- En el **Dataset A**, el R² debería ser alto (cercano a 1), y el MAE/RMSE relativamente bajos: la recta explica bien la relación entre horas y nota.
- En el **Dataset B**, el R² debería ser bajo (cercano a 0, o incluso negativo si el modelo predice peor que simplemente usar el promedio), y los errores serán proporcionalmente más altos respecto a la variabilidad de los datos: la recta no explica casi nada, porque no hay relación lineal real entre las variables.

**Punto didáctico central:** el R² y los errores no son solo números sueltos — son la evidencia cuantitativa de si el supuesto "a más horas, mejor nota" se sostiene o no en los datos. Un modelo puede "entrenarse" igual sobre cualquier dataset, pero eso no garantiza que la relación lineal exista.

---

## Qué conseguiste

- Entrenaste dos modelos de regresión lineal simple con `scikit-learn`, uno sobre datos con relación lineal clara y otro sin ella.
- Calculaste e interpretaste los residuos y las métricas de error (MAE, MSE, RMSE), entendiendo qué mide cada una y en qué unidades.
- Calculaste e interpretaste el R², comprendiendo qué significa que esté cerca de 1, cerca de 0, o sea negativo.
- Verificaste con evidencia numérica cómo se ve, en la práctica, la diferencia entre "hay correlación lineal" y "no la hay" — algo clave antes de asumir que un modelo lineal es apropiado para un problema real.
