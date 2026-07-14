# Clase Tres - 14 de Julio del 2026

# Repaso

* ML
  * Regresion
    * Algoritmos
      * Regresion lineal
    * Metricas
      * Error
  * Clasificacion
    * Algoritmos
      * Regresion Logistico
      * Basados en Arboles de Decision (Decission Tree)
      * KNN
    * Metricas
      * Matriz de Confusion
* Liberia de ML
  * MatPotLib
    * Seaborn
  * Pandas
  * Sckit-Learn
* Entornos de ML
  * Local VS Code
  * Google Colab
  * Azure ML Studio
  * AWS --> ???
  * Kaggle
* Proceso de ML
* Visualizacion de Datos
  * Histograma
    * Ejemplo de los dados (campana)
  * Plot (Lineas)
  * BoxPlot (Velas)
  * Scatter (puntitos)
  * HeatMap
* Matriz de Confusion

---

# Regresion

* Vamos a hacer un ejemplo de regresion

* En este caso vamos numpy, se puede instalar si no viene (aunque siempre viene instalado)

```python
!pip install numpy
```

* Defino el Dataset
```
import numpy as np

horas_estudio_a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9,10])
notas_a = np.array([1.5, 2.3, 2.5, 3.2, 5.5, 6, 7.3, 8.1, 8.5, 9.5])

horas_estudio_a
```

* Lo muestra

```
array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
```

> [!NOTE]
> En este caso use Numpy para que se manejen los arrays mas eficientemente
> El array asi como esta no funciona en scikit-learn hay que pasarlo de [1,2,3,4] a [ [1], [2], [3], [4] ]

* Para que podamos usar los features en sciit learn los tengo que tranformar con reshape

```
import numpy as np

horas_estudio_a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9,10])
notas_a = np.array([1.5, 2.3, 2.5, 3.2, 5.5, 6, 7.3, 8.1, 8.5, 9.5])

#OJO: 
print(horas_estudio_a)

horas_estudio_a = horas_estudio_a.reshape(-1, 1)

print(horas_estudio_a)
```

* Entrenamos un modelo de regresion Lineal

```
from sklearn.linear_model import LinearRegression

X = horas_estudio_a
y = notas_a

modelo_a = LinearRegression()
modelo_a.fit(X, y)

y_pred = modelo_a.predict(X)

print(y_pred)
print(y)
```

* Hago un gratico con matplotlib y la ayuda de la IA

```python
import matplotlib.pyplot as plt


# Gráfico
plt.figure(figsize=(8,5))

# Puntos reales
plt.scatter(X, y, color="blue", label="Datos reales")

# Línea de regresión
orden = np.argsort(X.flatten())
plt.plot(X.flatten()[orden],
         y_pred[orden],
         color="red",
         linewidth=2,
         label="Regresión lineal")

plt.xlabel("Horas de estudio")
plt.ylabel("Nota")
plt.title("Regresión Lineal: Horas de estudio vs Nota")
plt.legend()
plt.grid(True)

plt.show()
```

* Genero este grafico

<img width="686" height="471" alt="image" src="https://github.com/user-attachments/assets/537bcdd2-1da6-4947-8690-194183722063" />


## Metricas

* Cuatro metricas de error
 * MAE : Mean Absolute Error : El prompedio de el valor absoluto de los errores
 * MSE : Mean Square Error : A los errores los eleva al cuadrado y calcula el promedio (para castigar los errores mas grandes)
 * RMSE : LA raiz cuadrada del MSE (para mostrarlo en una magnitud comparable a los datos de entrada)
 * R2 : Mide el error en una magnitud que va entre 0 y 1


## Metricas Primer Caso

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae_a = mean_absolute_error(y, y_pred)
mse_a = mean_squared_error(y, y_pred)
rmse_a = np.sqrt(mse_a)
r2_a = r2_score(y, y_pred)

print(f"MAE: {mae_a:.4f}")
print(f"MSE: {mse_a:.4f}")
print(f"RMSE: {rmse_a:.4f}")
print(f"R2: {r2_a:.4f}")
```

### Otro ejemplo donde no haya relacion entre las variables

```
import numpy as np

horas_estudio_b = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9,10])
notas_b = np.array([6.5, 4.0, 7.8, 5.2, 6.9, 4.5, 7.1, 5.8, 6.0, 5.5])

#OJO: 
print(horas_estudio_b)

horas_estudio_b = horas_estudio_b.reshape(-1, 1)

print(horas_estudio_b)
```

* Voy a generar el segundo grafico

```python
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

X = horas_estudio_b
y = notas_b

modelo_b = LinearRegression()
modelo_b.fit(X, y)

y_pred = modelo_b.predict(X)

print(y_pred)
print(y)


# Gráfico
plt.figure(figsize=(8,5))

# Puntos reales
plt.scatter(X, y, color="blue", label="Datos reales")

# Línea de regresión
orden = np.argsort(X.flatten())
plt.plot(X.flatten()[orden],
         y_pred[orden],
         color="red",
         linewidth=2,
         label="Regresión lineal")

plt.xlabel("Horas de estudio")
plt.ylabel("Nota")
plt.title("Regresión Lineal: Horas de estudio vs Nota")
plt.legend()
plt.grid(True)

plt.show()
```

* Genero este grafico

<img width="691" height="471" alt="image" src="https://github.com/user-attachments/assets/f4f9281d-5030-47e6-acb5-3156b17a22de" />

* Calculo del error en el segundo caso

```
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae_a = mean_absolute_error(y, y_pred)
mse_a = mean_squared_error(y, y_pred)
rmse_a = np.sqrt(mse_a)
r2_a = r2_score(y, y_pred)

print(f"MAE: {mae_a:.4f}")
print(f"MSE: {mse_a:.4f}")
print(f"RMSE: {rmse_a:.4f}")
print(f"R2: {r2_a:.4f}")
```

* Para el primer ejemplo me dio

```
MAE: 0.3693
MSE: 0.1809
RMSE: 0.4254
R2: 0.9761
```

* Para el segundo ejemplo me dio

```
MAE: 0.9248
MSE: 1.2632
RMSE: 1.1239
R2: 0.0007
```

* Que ver
  * Cuando el MAE y el RMSE son parecidos es que no hay errores extremos

> [!NOTA]
> Una vez que tengo las metricas es una buena practica pedirle un analisis a la IA que me ayuda un monton

## Colinealidad

* Es cuando dos (o mas) variables predictoras estan fuertemente correlacionadas entre si. El r2 (coeficiente de determinacion) es cercano a 1.

## Reduccion de la dimensionalidad

---

# Pandas

* Vamos a hacer el mismo ejemplo con pandas

```python
import pandas as pd
import numpy as np

# Datos
df = pd.DataFrame({
    "Horas_Estudio": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Nota": [1.5, 2.3, 2.5, 3.2, 5.5, 6, 7.3, 8.1, 8.5, 9.5]
})

# Variables independientes (X) y dependiente (y)
X = df[["Horas_Estudio"]]   # Doble corchete => DataFrame
y = df["Nota"]              # Serie

```

---

# Clasificacion

## Metricas

