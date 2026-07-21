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
