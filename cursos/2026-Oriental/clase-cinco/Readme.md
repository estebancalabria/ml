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

> [!NOTE]
> Ver que en este dataset viene separado los features (california_housing.data) de los labels (california_housing.target)

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

* Graficos con MatPlotLi

* Histograma de los features

```
import matplotlib.pyplot as plt

X.hist(figsize=(10, 8))

plt.title("Distribucion de las variables")
plt.show()

```

* Histograma de los labels

```
import matplotlib.pyplot as plt

y.hist(figsize=(10, 8))

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

* Observaciones hay una correlacion logica ente Cantidad de Habitaciones y Cantidad de dormitorios

* Quisiera armar una variable de correlacion con el label
  
```
#Correlacion con el label

df_total = X.copy()
df_total["Value"] = y

corr_with_label = df_total.corr()["Value"].sort_values(ascending=False)

print(corr_with_label)
```

* Deteccion de outliners

```
# Deteccion de outliers con boxplot sin seaborn
plt.figure(figsize=(10, 8))

X.boxplot()
plt.show()
```

> [!NOTE]
> Me llama la atencion los outliners en la columna populatios

----

# Uso de la IA

* Como ya tengo una idea de como encarar el EDA puedo ayudarme mucho con la IA para ver como seguir y sacar conclusiones
* Respuesta de la IA
 
Esteban, el EDA está bien encaminado para una clase introductoria porque ya cubre carga del dataset, estructura, nulos, descriptivos, histogramas, correlaciones y outliers. Pero como análisis exploratorio todavía está “a mitad de camino”: muestra cosas importantes, aunque faltan varias comprobaciones para transformar esas observaciones en decisiones de modelado.

## 1. Conclusiones que sacaría del EDA actual

### 1. El dataset está limpio en cuanto a nulos

En `California Housing` tenés:

```python
X.isnull().sum()
y.isnull().sum()
```

y todas las variables dan `0` nulos. Eso permite avanzar sin imputación inicial. Para una clase está perfecto remarcar:

> “No siempre un EDA termina en limpieza. A veces confirma que algunas tareas no son necesarias.”

Igual, ojo: que no haya nulos no significa que el dataset esté “perfecto”. Todavía hay que mirar outliers, escalas, distribuciones y relaciones con la variable objetivo.

***

### 2. Todas las variables son numéricas

`X.info()` muestra 8 columnas `float64`. Eso simplifica bastante el pipeline porque no hace falta codificación de variables categóricas.

Variables:

```text
MedInc
HouseAge
AveRooms
AveBedrms
Population
AveOccup
Latitude
Longitude
```

La variable objetivo es:

```text
Value
```

Esto es importante porque permite usar directamente modelos de regresión, aunque algunos modelos van a requerir escalamiento.

***

### 3. El dataset tiene buen tamaño

El dataset tiene:

```text
20640 filas × 8 columnas
```

Es un tamaño razonable para entrenar, validar y hacer gráficos sin problemas. Para una práctica de alumnos está muy bien porque no es un dataset mínimo artificial, pero tampoco es inmanejable.

***

### 4. Hay variables con distribuciones muy sesgadas

Los histogramas muestran que varias variables tienen colas largas:

* `MedInc` está concentrada en valores bajos/medios y tiene cola hacia valores altos.
* `AveRooms` tiene valores extremos muy altos.
* `AveBedrms` tiene la mayoría de valores cerca de 1, pero algunos casos extremos.
* `Population` tiene una cola larga muy marcada.
* `AveOccup` tiene un outlier enorme, con máximo de `1243.333333`.

Esto se ve también en `describe()`:

```text
AveOccup:
mean = 3.07
median = 2.81
max = 1243.33
```

Ese máximo es brutalmente superior al percentil 75:

```text
75% = 3.28
max = 1243.33
```

Conclusión fuerte:

> Hay outliers importantes, especialmente en `AveOccup`, `Population`, `AveRooms` y `AveBedrms`.

Para modelos lineales, esos outliers pueden afectar mucho. Para árboles o random forest, el impacto suele ser menor.

***

### 5. La variable más relacionada con el precio es `MedInc`

La correlación con el label muestra:

```text
MedInc      0.688075
AveRooms    0.151948
HouseAge    0.105623
AveOccup   -0.023737
Population -0.024650
Longitude  -0.045967
AveBedrms  -0.046701
Latitude   -0.144160
```

Conclusión principal:

> `MedInc`, ingreso medio, es claramente la variable con mayor relación lineal positiva con `Value`.

Esto tiene sentido: zonas con mayor ingreso medio tienden a tener casas de mayor valor.

Pero también hay que explicarlo bien a los alumnos:

* Correlación alta no implica causalidad.
* Correlación lineal no detecta relaciones no lineales.
* Una variable con baja correlación individual puede ser útil combinada con otras.

Por ejemplo, `Latitude` y `Longitude` tienen correlaciones simples bajas/moderadas con el precio, pero geográficamente pueden ser muy importantes.

***

### 6. Hay relaciones fuertes entre algunas variables predictoras

La matriz de correlación sugiere:

* `AveRooms` y `AveBedrms` tienen correlación positiva fuerte.
* `Latitude` y `Longitude` tienen correlación negativa fuerte.
* `MedInc` parece tener cierta relación positiva con `AveRooms`.

Esto puede importar para modelos lineales porque puede haber multicolinealidad. En modelos de árboles suele preocupar menos.

Una conclusión didáctica útil:

> No alcanza con mirar la relación de cada variable con el label. También hay que mirar la relación entre las variables predictoras.

***

### 7. El boxplot general no es muy legible

El boxplot de todas las columnas juntas queda dominado por `Population`, porque está en una escala muchísimo más grande que el resto. Visualmente “aplasta” a las demás variables.

Esto no significa que el gráfico esté mal, pero sí que no es el mejor gráfico para comparar outliers de variables con escalas tan distintas.

Conclusión:

> El boxplot global sirve para detectar que hay escalas muy diferentes, pero no sirve bien para analizar cada variable.

Mejor hacer boxplots separados o escalar antes solo para visualizar.

***

## 2. Cosas que están bien del EDA

El flujo que armaste tiene varios puntos correctos:

### Correcto

```python
X.head()
X.info()
X.describe()
X.isnull().sum()
X.hist()
X.corr()
correlacion con el label
boxplot
```

Eso forma una muy buena primera plantilla de EDA.

Para una clase, me gusta porque muestra esta progresión:

1. Ver datos.
2. Entender estructura.
3. Revisar nulos.
4. Mirar estadísticos.
5. Visualizar distribuciones.
6. Analizar correlaciones.
7. Detectar outliers.
8. Pensar impacto en modelos.

Está muy bien como esqueleto.

***

## 3. Qué falta o qué mejoraría

### 1. Falta incluir `y` en el análisis descriptivo

Ahora analizás muy bien `X`, pero el label `Value` queda menos explorado. Hiciste histograma de `y`, pero faltaría:

```python
print(y.describe())
```

Y también podrías ver si `Value` tiene valores máximos repetidos, porque en California Housing el target está topado en algunos valores altos. Eso puede impactar el modelo.

Código:

```python
print(y.describe())

print("Valores máximos más frecuentes:")
print(y["Value"].value_counts().sort_index(ascending=False).head(10))
```

Conclusión buscada:

> Si muchos valores están acumulados en el máximo, el modelo puede tener dificultad para predecir correctamente viviendas caras porque el target está censurado o limitado.

***

### 2. Falta analizar outliers variable por variable

El boxplot conjunto no alcanza. Haría esto:

```python
import matplotlib.pyplot as plt

for col in X.columns:
    plt.figure(figsize=(6, 4))
    X.boxplot(column=col)
    plt.title(f"Boxplot de {col}")
    plt.show()
```

Y también contaría outliers con IQR:

```python
for col in X.columns:
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1

    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    outliers = X[(X[col] < limite_inferior) | (X[col] > limite_superior)]

    print(col)
    print("Cantidad de outliers:", len(outliers))
    print("Porcentaje:", round(len(outliers) / len(X) * 100, 2), "%")
    print("-------------------------")
```

Esto ya convierte la observación visual en algo cuantificado.

***

### 3. Falta analizar relación entre cada variable y el target

La correlación ayuda, pero conviene ver scatterplots.

```python
import matplotlib.pyplot as plt

df_total = X.copy()
df_total["Value"] = y

for col in X.columns:
    plt.figure(figsize=(6, 4))
    plt.scatter(df_total[col], df_total["Value"], alpha=0.3)
    plt.xlabel(col)
    plt.ylabel("Value")
    plt.title(f"{col} vs Value")
    plt.show()
```

Esto es clave porque podés ver:

* Relaciones lineales.
* Relaciones no lineales.
* Outliers.
* Saturaciones del target.
* Variables que parecen no explicar nada individualmente.

Para este dataset, `MedInc vs Value` suele ser el gráfico más claro.

***

### 4. Falta hacer análisis geográfico

Como tenés `Latitude` y `Longitude`, sería muy interesante hacer un scatter geográfico:

```python
plt.figure(figsize=(8, 6))
plt.scatter(
    X["Longitude"],
    X["Latitude"],
    c=y["Value"],
    cmap="viridis",
    alpha=0.4
)
plt.colorbar(label="Value")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Ubicación geográfica coloreada por valor de vivienda")
plt.show()
```

Esto es muy potente didácticamente porque muestra que una variable puede no tener alta correlación lineal de forma individual, pero visualmente sí aportar mucha información espacial.

Conclusión esperable:

> La ubicación geográfica parece relevante para el valor de la vivienda, aunque `Latitude` y `Longitude` no aparezcan como las variables más correlacionadas individualmente.

***

### 5. Falta revisar escalas antes de modelar

Las variables están en escalas muy distintas:

* `MedInc`: aproximadamente 0.5 a 15.
* `HouseAge`: 1 a 52.
* `Population`: 3 a 35682.
* `Latitude`: 32.54 a 41.95.
* `Longitude`: -124.35 a -114.31.

Esto no afecta igual a todos los modelos.

Conviene explicar:

| Modelo              | ¿Necesita escalamiento? |
| ------------------- | ----------------------- |
| Regresión lineal    | Recomendable            |
| Regresión logística | Recomendable            |
| KNN                 | Sí, muy importante      |
| SVM                 | Sí, muy importante      |
| Árbol de decisión   | No suele ser necesario  |
| Random Forest       | No suele ser necesario  |
| Gradient Boosting   | No suele ser necesario  |

En tu ejemplo anterior de regresión logística con 5 filas, diría algo importante: no se puede concluir que “con escalamiento es más preciso” porque en ese mini dataset el modelo escalado dio una matriz peor en entrenamiento:

Sin escalar:

```text
[[3 0]
 [1 1]]
```

Escalado:

```text
[[2 1]
 [1 1]]
```

Si asumimos matriz con clases `[0, 1]`, sin escalar acierta 4 de 5, escalado acierta 3 de 5.

Entonces yo cambiaría el comentario:

```python
# Ejemplo sin escalamiento
```

y

```python
# Mismo ejemplo con escalamiento
# El escalamiento ayuda a que las variables estén en rangos comparables,
# pero no garantiza mejor resultado en todos los casos.
```

Mucho más correcto.

***

### 6. Falta separar entrenamiento y prueba antes del escalamiento

Este punto es importante para buenas prácticas.

Si vas a escalar para modelar, no conviene hacer:

```python
scaler.fit_transform(X)
```

antes de dividir. Eso mete información del conjunto completo en el escalador.

Mejor:

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Conclusión:

> El escalador se ajusta solo con entrenamiento y luego se aplica al test.

Esto evita data leakage.

***

### 7. Falta una matriz de correlación con valores visibles

La matriz con `imshow` está bien, pero falta imprimir la matriz o anotar valores. Para alumnos, ver colores sin números puede ser confuso.

Sin seaborn podrías hacer:

```python
corr = df_total.corr()

plt.figure(figsize=(10, 8))
plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
plt.colorbar()
plt.title("Matriz de correlación")

plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)

for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}",
                 ha="center", va="center", color="black")

plt.show()
```

Esto mejora muchísimo la interpretación.

***

### 8. Falta ver distribuciones transformadas

Como hay variables muy sesgadas, podrías probar escala logarítmica en algunas:

```python
import numpy as np

cols_sesgadas = ["Population", "AveOccup", "AveRooms", "AveBedrms"]

for col in cols_sesgadas:
    plt.figure(figsize=(6, 4))
    plt.hist(np.log1p(X[col]), bins=30)
    plt.title(f"Distribución logarítmica de {col}")
    plt.xlabel(f"log1p({col})")
    plt.ylabel("Frecuencia")
    plt.show()
```

Esto ayuda a explicar que a veces no eliminamos outliers directamente, sino que transformamos variables.

***

## 4. Exploraciones adicionales que haría

### A. Ranking de correlaciones absolutas con el target

```python
corr_with_label = df_total.corr()["Value"].drop("Value")
corr_abs = corr_with_label.abs().sort_values(ascending=False)

print(corr_abs)
```

Esto permite decir:

> “Estas son las variables más asociadas linealmente al precio, sin importar si la relación es positiva o negativa.”

***

### B. Pairplot manual simplificado

Como no querés usar seaborn, podés hacer algunos scatterplots clave:

```python
variables_clave = ["MedInc", "AveRooms", "HouseAge", "Latitude", "Longitude"]

for col in variables_clave:
    plt.figure(figsize=(6, 4))
    plt.scatter(X[col], y["Value"], alpha=0.3)
    plt.xlabel(col)
    plt.ylabel("Value")
    plt.title(f"Relación entre {col} y Value")
    plt.show()
```

***

### C. Comparar barrios/zonas de alto y bajo valor

Podés crear grupos por precio:

```python
df_total["grupo_precio"] = pd.qcut(
    df_total["Value"],
    q=4,
    labels=["Bajo", "Medio-bajo", "Medio-alto", "Alto"]
)

print(df_total.groupby("grupo_precio").mean())
```

Esto es excelente para clase porque traduce números a comparación:

> “¿Cómo son las zonas de precio alto frente a las zonas de precio bajo?”

***

### D. Ver si hay duplicados

```python
print("Duplicados:", X.duplicated().sum())
```

No es lo más importante en este dataset, pero como hábito de EDA está bueno.

***

### E. Ver asimetría de variables

```python
print(X.skew().sort_values(ascending=False))
```

Esto te permite detectar variables muy sesgadas sin depender solo del histograma.

***

### F. Evaluar baseline de modelo

Después del EDA, haría un baseline simple:

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

X_train, X_test, y_train, y_test = train_test_split(
    X, y["Value"],
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2:", r2_score(y_test, y_pred))
```

Esto conecta el EDA con modelado.

***

## 5. Qué corregiría del notebook

### Corrección 1: error conceptual en escalamiento

Donde dice:

```python
# Ejemplo sin escalamiento de datos (Mas impreciso)
```

yo no diría eso. Mejor:

```python
# Ejemplo sin escalamiento de datos
```

Y donde dice:

```python
# Mismo ejemplo pero con escalamiento de datos (Mas preciso)
```

mejor:

```python
# Mismo ejemplo con escalamiento de datos
# Escalar puede ayudar a ciertos modelos, pero no garantiza mayor precisión.
```

Especialmente porque con tu matriz de confusión el modelo escalado no fue mejor.

***

### Corrección 2: dataset demasiado chico para sacar conclusiones

El ejemplo de edad, sueldo y aprueba tiene solo 5 filas. Sirve para explicar mecánica, pero no para evaluar precisión real.

Diría en clase:

> “Este dataset mínimo sirve para entender el proceso, pero no para sacar conclusiones estadísticas ni comparar modelos de forma seria.”

***

### Corrección 3: train/test con 5 filas no es representativo

Con:

```python
test_size=0.2
```

el test tiene 1 sola fila. Eso está bien para mostrar cómo funciona `train_test_split`, pero no para evaluar modelos.

Podés aclarar:

> “Con datasets tan pequeños, la separación train/test no permite una evaluación confiable.”

***

### Corrección 4: falta usar `stratify` en clasificación

En el ejemplo de clasificación `aprueba`, si quisieras dividir en train/test, convendría:

```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

Aunque con 5 filas puede fallar o no ser útil por el tamaño mínimo.

***

## 6. Mi diagnóstico general

El EDA está bien como primera versión didáctica. Logra mostrar lo básico: estructura, nulos, estadísticos, histogramas, correlaciones y outliers. La conclusión más fuerte es que `MedInc` es la variable más asociada al valor de las viviendas, mientras que varias variables tienen outliers importantes y distribuciones muy sesgadas, especialmente `AveOccup`, `Population`, `AveRooms` y `AveBedrms`.

Lo que falta para que quede más sólido es analizar mejor el target `Value`, visualizar la relación de cada variable contra el precio, revisar outliers de forma individual y cuantificada, incorporar análisis geográfico con `Latitude` y `Longitude`, y conectar el EDA con decisiones concretas de modelado: escalamiento, transformación logarítmica, tratamiento de outliers y elección de modelos.

## 7. Versión corta para decirlo en clase

Podrías explicarlo así:

> “El dataset no tiene valores nulos y todas las variables son numéricas, así que podemos avanzar rápido al análisis exploratorio. Vemos que algunas variables tienen escalas muy diferentes y distribuciones sesgadas, especialmente `Population`, `AveOccup`, `AveRooms` y `AveBedrms`, donde aparecen outliers importantes. La variable que más se relaciona linealmente con el valor de la vivienda es `MedInc`, con una correlación aproximada de 0.69. Sin embargo, otras variables como latitud y longitud pueden aportar información geográfica aunque su correlación individual no sea tan alta. Antes de modelar, conviene analizar mejor el target, revisar outliers por variable, hacer gráficos de dispersión contra `Value`, considerar transformaciones logarítmicas y separar correctamente entrenamiento y prueba para evitar data leakage.”

## 8. Código que agregaría al EDA

Te dejo un bloque compacto para completar el notebook:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Unimos X e y para análisis conjunto
df_total = X.copy()
df_total["Value"] = y["Value"]

# 1. Análisis del target
print("Descripción de la variable objetivo:")
print(df_total["Value"].describe())

plt.figure(figsize=(6, 4))
plt.hist(df_total["Value"], bins=30)
plt.title("Distribución de Value")
plt.xlabel("Value")
plt.ylabel("Frecuencia")
plt.show()

# 2. Correlación con el target
corr_with_label = df_total.corr()["Value"].sort_values(ascending=False)
print("Correlación con Value:")
print(corr_with_label)

# 3. Ranking de correlaciones absolutas
print("Ranking por correlación absoluta con Value:")
print(corr_with_label.drop("Value").abs().sort_values(ascending=False))

# 4. Scatterplots contra el target
for col in X.columns:
    plt.figure(figsize=(6, 4))
    plt.scatter(df_total[col], df_total["Value"], alpha=0.3)
    plt.xlabel(col)
    plt.ylabel("Value")
    plt.title(f"{col} vs Value")
    plt.show()

# 5. Outliers por IQR
for col in X.columns:
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1

    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    outliers = X[(X[col] < limite_inferior) | (X[col] > limite_superior)]

    print(col)
    print("Límite inferior:", limite_inferior)
    print("Límite superior:", limite_superior)
    print("Cantidad de outliers:", len(outliers))
    print("Porcentaje:", round(len(outliers) / len(X) * 100, 2), "%")
    print("--------------------------------")

# 6. Boxplots individuales
for col in X.columns:
    plt.figure(figsize=(6, 4))
    X.boxplot(column=col)
    plt.title(f"Boxplot de {col}")
    plt.show()

# 7. Mapa geográfico coloreado por Value
plt.figure(figsize=(8, 6))
plt.scatter(
    df_total["Longitude"],
    df_total["Latitude"],
    c=df_total["Value"],
    cmap="viridis",
    alpha=0.4
)
plt.colorbar(label="Value")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Ubicación geográfica coloreada por valor de vivienda")
plt.show()

# 8. Asimetría
print("Asimetría de las variables:")
print(X.skew().sort_values(ascending=False))

# 9. Transformación logarítmica para variables sesgadas
cols_sesgadas = ["Population", "AveOccup", "AveRooms", "AveBedrms"]

for col in cols_sesgadas:
    plt.figure(figsize=(6, 4))
    plt.hist(np.log1p(X[col]), bins=30)
    plt.title(f"Distribución logarítmica de {col}")
    plt.xlabel(f"log1p({col})")
    plt.ylabel("Frecuencia")
    plt.show()
```

Mi recomendación final: para clase, dejaría el EDA como “primer EDA básico” y después agregaría una sección llamada **“EDA ampliado orientado a modelado”**, donde metas outliers, relación contra `Value`, mapa geográfico, transformaciones y baseline de regresión. Ahí el notebook queda mucho más redondo.

---

# Entrenando un modelo KNN

* Sin Normalizar las variables
  
```python
# Entrenar sin normalizar
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo_knn_sin_escalar = KNeighborsRegressor(n_neighbors=5)
modelo_knn_sin_escalar.fit(X_train, y_train)

y_pred_sin_escalar = modelo_knn_sin_escalar.predict(X_test)

mae_sin_escalar = mean_absolute_error(y_test, y_pred_sin_escalar)
r2_sin_escalar = r2_score(y_test, y_pred_sin_escalar)

print("MAE sin escalar:", mae_sin_escalar)
print("R2 sin escalar:", r2_sin_escalar)
```

* Ahora normalizando variables

```
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Escalar
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar KNN
modelo_knn_escalado = KNeighborsRegressor(n_neighbors=5)

modelo_knn_escalado.fit(X_train_scaled, y_train)

# Predicción
y_pred_escalado = modelo_knn_escalado.predict(X_test_scaled)

# Métricas
mae_escalado = mean_absolute_error(y_test, y_pred_escalado)
r2_escalado = r2_score(y_test, y_pred_escalado)

print("MAE con StandardScaler:", mae_escalado)
print("R2 con StandardScaler:", r2_escalado)
```

