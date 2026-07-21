# Clase Cinco - 21 de Julio del 2026

# Repaso

* Metricas
  * Regresion
    * MAE
    * MSE
    * RMSE
    * R2
  * Clasificacion
    * Matriz de Confusion
    * Accuracy/Precision
    * Recall
    * Specificty
* Modelos conservadores vs Alarmistas
* Pandas
  * EDA (Exploracion de Datos)
  * Limpieza de Datos
* Preparacion de datos
  * Seleccion de features o variables predictoras
  * Encoding y codificacion de variables
    * OrdinalEncoder
    * OneHotEncoder
* Ejemplo integrador
    * Encoding
    * DecisionTreeClassifier
      * Alta Explicabilidad
     
---

# Algoritmos de ML

* Basados en Arboles
  * DecisionTreeClassifier  << Baseline
  * Gradient Booted Trees
  * Random Forrest
  * XGBoost
* Basados en Distancia
  * Regresion Lineal (Regresion)  <<  Baseline
  * Regresion Logistica (Clasificacion)
  * KNN
  * SVM
  * Red Neuronal
  * ...

---

# Normalizacion de Variables

* Tengo dos variables en mi dataser
  * Altura : Entre 50 y 250
  * Peso : Entre 50 y 250
  * Largo : 1000000000 y 100000000000000
* La dimension impacta mucho para los algoritmos de distancia.
* Valores muy grandes opacan a valores chicos, tenemos que llevar todo a una magnitud comun.
* En general se lleva todo a una magnitud entre 0 y 1.

> [NOTE!]
> Los algoritmos basados en distancia requiren de normalizacion de variables

## StandardScaler

```python

import pandas as pd

df = pd.DataFrame({
    "edad" : [20, 21, 39, 63, 45],
    "sueldo" : [1000, 2000, 3000, 4000, 5000],
})

df

```

* Escalando con StandardScaller (Se usa en el 90% de los casos)

```
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df_scaled = pd.DataFrame(
    scaler.fit_transform(df),
    columns = df.columns
)


df_scaled
```

* Funcionamiento
 * Calculamos la media y esta va a ser nuesto 0
 * Calculamos el desvio estandar : este va a ser 1 

* Es senible a outlines

```
import pandas as pd

df = pd.DataFrame({
    "edad" : [20, 21, 39, 63, 4500],
    "sueldo" : [1000, 2000, 3000, 4000, 5000],
})

df
```

> [NOTE!]
> Una edad de 4500 anios es claramente un dato que esta mal que se conoce como outliner
> Parte de la exploracion y preparacion de datos consisten en detectar y eliminar esos outliners para que los algotimos basados en distancia trabajemn mejor

## Min Max Scaller

```
from sklearn.preprocessing import MinMaxScaler 

scaler = MinMaxScaler()

df_min_max_scaled = pd.DataFrame(
    scaler.fit_transform(df),
    columns = df.columns
)

df_min_max_scaled
```

* El valor mas chico -> 0
* Valor mas grande -> 1


# Comparacion con o sin Escalado

* Ejemplo si escalamiento
```
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

df = pd.DataFrame({
    "edad" : [20, 21, 39, 63, 45],
    "sueldo" : [1000, 2000, 3000, 4000, 5000],
    "aprueba" : [1, 0, 1, 0, 0]
})

#Uso Regresion Logistica
#Ejemplo sin ecalamiento de datos (Mas impreciso)


X = df[["edad", "sueldo"]]
y = df["aprueba"]

model = LogisticRegression()
model.fit(X, y)

y_pred = model.predict(X)

#Matriz de confusion
cm = confusion_matrix(y, y_pred)

print(cm)
```

> [!NOTE]
> El ejemplo anterior para un algotritmo basado en distancia como es la regresion logisitica no es confiable, el sueldo pesa mucho mas que la edad

* Con Escalamiento
```
#Mismo ejemplo pero con escalamiento de datos (Mas preciso)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

model_scaled = LogisticRegression()
model_scaled.fit(X_scaled, y)

y_pred_scaled = model_scaled.predict(X_scaled)

#Matriz de confusion
cm_scaled = confusion_matrix(y, y_pred_scaled)
print(cm_scaled)
```

# Cross Validation

* Cuando quiero validar un modelo no es recomendable utilizar los mismos datos de entrenaminto que estuvimos usando
* De esa manera si hay overfitting (el modelo memoriza los datos de entrenamiento) podemos pensar que tenemos un modelo muy bueno cuando en realidad es un memorizador
* Por eso esta bueno que nuestros datasets tengan mas +100 (o + 1000 o +1M)
* Vamos a dividir los datos en dos
 * Datos de Entrenamiento
 * Datos de Validacion ( para calcular las metricas del modelo)

> [!NOTE]
> Hasta ahora como estuvimos trabajando con datasets muy chicos utilizamos los datos de entrenamiento para validar
> En la practica se dividen, y se valida el modelo con datos que no conocio en el entrenamiento

* Estandar de division
 * 80% datos -> Etrenamiento
 * 20% datos -> Validacion

* El proceso de reentrenar un modelo con distintas divisiones entre datos de entrenamiento y  validacion se llama : cross validation


```
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.DataFrame({
    "edad" : [20, 21, 39, 63, 45],
    "sueldo" : [1000, 2000, 3000, 4000, 5000],
    "aprueba" : [1, 0, 1, 0, 0]
})

#Dividimos el dataset en entrenamiento y prueba


X = df[["edad", "sueldo"]]
y = df["aprueba"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X")
print(X)
print("X_train:")
print(X_train)
``` 

* Para hacer cross validation vamos variando los valores de
 * random_state : Para que me de una division distinta
 * test_size : Para variar el tamanio entre la muestra e entreamiento y validacion

---

# Ejemplo Integrador

* Vamos a utilizar el dataset de California Housing

```python
#California Housing
from sklearn.datasets import fetch_california_housing
import pandas as pd

california_housing = fetch_california_housing()

df = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)

df
```

* Vamos a hacer la exploracion de datos

```
print(df.shape)

X = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
y = pd.DataFrame(california_housing.target, columns=["Value"])

#Hay nulos en el dataset?
print(X.isnull().sum())
print(y.isnull().sum())

#EDA...
```

* Exploracion de datos con pandas

```
print("Ver las primeras filas")
print(X.head())
print("---------------------------------")

print("Informacion del dataset")
print(X.info())
print("---------------------------------")

print("Descripcion del dataset")
print(X.describe())
print("---------------------------------")
```

* Graficos con MatPlotLib (Histograma)

```
import matplotlib.pyplot as plt

X.hist(figsize=(10, 8))

plt.title("Distribucion de las variables")
plt.show()

```

* Matriz de Correlacion

```python
# Correlacion entre variables sin seaborn
import matplotlib.pyplot as plt

corr = X.corr()

plt.figure(figsize=(10, 8))
plt.imshow(corr, cmap='coolwarm', interpolation='nearest')
plt.colorbar()


plt.title("Matriz de correlacion")

plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)

plt.show()
```
