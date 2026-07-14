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
* Si hay colinealidd entre nuetra entrada (features) y el target (lo que queremos predecir), todo bien, excelente
  * El problema es cuando dos variables de entrada (features) son colineales, las dos aportan la misma informacion (o parecida) 

### Matriz de Correlacion

* Calcula la regresion lineal (el r2) entre todos los pares de variables y estudia colinealidad
* Es muy util en la exploracion de datos
* Hay dos formas de calcular la matriz de correlacion
  * Numpy -> np.corrcoef(matriz_datos)
  * Pandas -> df.corr()
 
```python
horas_estudio = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9,10])
asistencia = np.array([92, 55, 98, 60, 85, 65, 90, 58, 95, 70])  
notas = np.array([1.5, 2.3, 2.5, 3.2, 5.5, 6, 7.3, 8.1, 8.5, 9.5])
```

* Calculamos la matriz de correlacion con numpy

```
import numpy as np

nombres = ["Horas de Estudio", "Asistencia", "Notas"]
matriz_datos =  np.array([horas_estudio, asistencia, notas])

print(matriz_datos)

matriz_correlacion_numpy = np.corrcoef(matriz_datos)

print("Matriz de correlación:")
print(matriz_correlacion_numpy)

```

* Salida

```
[[ 1.   2.   3.   4.   5.   6.   7.   8.   9.  10. ]
 [92.  55.  98.  60.  85.  65.  90.  58.  95.  70. ]
 [ 1.5  2.3  2.5  3.2  5.5  6.   7.3  8.1  8.5  9.5]]
Matriz de correlación:
[[ 1.         -0.0523297   0.98798407]
 [-0.0523297   1.         -0.04464665]
 [ 0.98798407 -0.04464665  1.        ]]
```

#### Graficando matriz de correlacion

```
plt.figure(figsize=(6,5))

plt.imshow(matriz_correlacion_numpy,
           cmap="coolwarm",
           vmin=-1,
           vmax=1)

plt.colorbar(label="Correlación")

plt.xticks(range(len(nombres)), nombres, rotation=45)
plt.yticks(range(len(nombres)), nombres)

# Mostrar el valor en cada celda
for i in range(len(nombres)):
    for j in range(len(nombres)):
        plt.text(j, i,
                 f"{matriz_correlacion_numpy[i, j]:.2f}",
                 ha="center",
                 va="center",
                 color="black")

plt.title("Heatmap de Correlaciones")
plt.tight_layout()
plt.show()
```

* Me genero este grafico

<img width="595" height="482" alt="image" src="https://github.com/user-attachments/assets/f930ea9f-474a-44a0-a781-4fe21fa95d53" />

### Matriz de Correlacion con Pandas

```
import pandas as pd

df = pd.DataFrame({
    "Horas de Estudio": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Asistencia": [92, 55, 98, 60, 85, 65, 90, 58, 95, 70],
    "Notas": [1.5, 2.3, 2.5, 3.2, 5.5, 6, 7.3, 8.1, 8.5, 9.5]
})

matriz_correlacion = df.corr()

print(matriz_correlacion)
```

* Salida

```
Horas de Estudio  Asistencia     Notas
Horas de Estudio          1.000000   -0.052330  0.987984
Asistencia               -0.052330    1.000000 -0.044647
Notas                     0.987984   -0.044647  1.000000
```

* Graficando el heatmap

```
plt.figure(figsize=(6,5))

plt.imshow(matriz_correlacion,
           cmap="coolwarm",
           vmin=-1,
           vmax=1)

plt.colorbar(label="Correlación")

plt.xticks(range(len(matriz_correlacion.columns)),
           matriz_correlacion.columns,
           rotation=45)

plt.yticks(range(len(matriz_correlacion.columns)),
           matriz_correlacion.columns)

for i in range(len(matriz_correlacion)):
    for j in range(len(matriz_correlacion)):
        plt.text(j, i,
                 f"{matriz_correlacion.iloc[i, j]:.2f}",
                 ha="center",
                 va="center")

plt.title("Heatmap de Correlaciones")
plt.tight_layout()
plt.show()
```

* Grafico Generado

<img width="595" height="482" alt="image" src="https://github.com/user-attachments/assets/02f1d982-94c9-4ad6-9be6-800e18fba789" />

## Reduccion de la dimensionalidad

* Cuando dos features (variables que uso para predecir) son colineales, a veces no esta tan bueno
  * Baja la explicabilidad, el modelo se confunde y marea con las dos variables
  * Hay modelos que no funcionan de manera optima si hay colinealidad
    * Los modelos basados en arboles -> Zafan, puteden funcionar, no se penalizan tanto.
      * Random Forrest, Decission Tree, XGBoost
    * Lineales y basado en distancia -> Van para atras, no se llevan muy bien con la colinealidad
      * KNN, SVN, .... 

* Si usamos como features dos variables colineales
  * Algunos modelos no andan tan bien
  * Perdemos poder de  explicacion (ver importancia de variables)
  * Tardan mas...
 
* La reduccion de la dimensionalidad consiste en elegir que columnas realmente son importantes/relevantes a la hora entrenar el modelo
* Muchas veces no conviene usar todas las columnas

---

# Pandas

* Nuestros datos de entrada en ML los podemos representar como
  * Array de Python (poco eficiente)
  * Array de Numpy
  * Dataframe de Pandas

* Vamos a hacer el mismo ejemplo con pandas

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# Datos
df = pd.DataFrame({
    "Horas_Estudio": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Nota": [1.5, 2.3, 2.5, 3.2, 5.5, 6, 7.3, 8.1, 8.5, 9.5]
})

# Variables independientes (X) y dependiente (y)
X = df[["Horas_Estudio"]]   # Doble corchete => DataFrame
y = df["Nota"]              # Serie

modelo = LinearRegression()
modelo.fit(X, y)

y_pred = modelo.predict(X)

plt.figure(figsize=(8,5))

# Puntos reales
plt.scatter(X, y, color="blue", label="Datos reales")

# Línea de regresión
orden = np.argsort(X.values.flatten())

plt.plot(
    X.values.flatten()[orden],
    y_pred[orden],
    color="red",
    linewidth=2,
    label="Regresión lineal"
)

plt.xlabel("Horas de estudio")
plt.ylabel("Nota")
plt.title("Regresión Lineal: Horas de estudio vs Nota")
plt.legend()
plt.grid(True)

plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

```

## Trabajo con archivos csv

* Lo levanto localmente
```python
import pandas as pd

df = pd.read_csv("datos_estudio_asistencia_notas.csv")

df.head()
```

* O lo levanto desde Internet

```python
import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/estebancalabria/ml/refs/heads/main/cursos/2026-Oriental/clase-tres/datos_estudio_asistencia_notas.csv")

df.head()
```

* Hacemos lo mismo que antes

```python
matriz_correlacion = df.corr()

plt.figure(figsize=(6,5))

plt.imshow(matriz_correlacion,
           cmap="coolwarm",
           vmin=-1,
           vmax=1)

plt.colorbar(label="Correlación")

plt.xticks(range(len(matriz_correlacion.columns)),
           matriz_correlacion.columns,
           rotation=45)

plt.yticks(range(len(matriz_correlacion.columns)),
           matriz_correlacion.columns)

for i in range(len(matriz_correlacion)):
    for j in range(len(matriz_correlacion)):
        plt.text(j, i,
                 f"{matriz_correlacion.iloc[i, j]:.2f}",
                 ha="center",
                 va="center")

plt.title("Heatmap de Correlaciones")
plt.tight_layout()
plt.show()
```

* Grafico Generado

<img width="595" height="482" alt="image" src="https://github.com/user-attachments/assets/e079951d-c0c5-40c0-9c72-22a86cdea99c" />

## Probar con otro dataset...

* Quiero ver si hay alguna correlacion entre la posicion del ranking de la fifa de estos paises
```
import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/estebancalabria/ml/refs/heads/main/datasets/fifa.csv")

#Descartar primer columna de fecha
df = df.iloc[:, 1:]

corr = df.corr()
```

* Conclusion : Muy poca correlacion

---

# Clasificacion

## Metricas

