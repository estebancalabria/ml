# Clase Cuatro - 16 de Julio del 2026

# Repaso 

* Formas de Definir el Data Set
  * Arrays De Python
  * Numpy
  * Pandas
    * DataFrame
    * Levantar desde un CSV
* Metricas
  * Regresion
    * MAE : Mean Absolute Error
    * Error elevados al cuadrado (Castiga errores altos)
      * MSE : Mean Square Error
      * RMSE : Root Mean Square Root
    * R2 -> [0 y 1] -> 1 Son completamente colineales, 0 no tiene nada que ver una con la otra
      * Explica que tan bueno es el modelo para predecir los datos
      * R2 > 0.8 ---> Las variables estan relacionadas
  * Clasificacion
    * Matriz de Confusion
    * Accuracy
    * Precision
    * Recall
    * Specificity
* Colinealidad / Correlacion
  * Matriz de Correlacion
  * Reduccion de la dimensionalidad (Feature Seleccion)

 ---

 # Laboratorio Preccission vs Recall

* Definir mi Dataset
  
```python
import numpy as np

# Es un banco que trata de predecir transaccions fraudulentas
# 0 -> Transaccion Normal   1-> Transaccion Fraudulenta

y_realidad = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])

# Modlelo a : "Conservador" : Marca fraude cuando esta muy seguro
y_pred_a   = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])

# Modelo B : "Alarmista" ->  trata de no dejar pasar un posible fraude
y_pred_b   = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0 ,0 ,1, 0, 0 ,0 ,0, 1, 1, 1, 0])
```

* Calcular metricas para modelos a y modelo b

```
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)

# Modelo A
accuracy_a = accuracy_score(y_realidad, y_pred_a)
precision_a = precision_score(y_realidad, y_pred_a)
recall_a = recall_score(y_realidad, y_pred_a)
confusion_matrix_a = confusion_matrix(y_realidad, y_pred_a)

print ("Modelo A")
print ("Accuracy:", accuracy_a)
print ("Precision:", precision_a)
print ("Recall:", recall_a)
print ("Confusion Matrix:\n", confusion_matrix_a)

print("------------------------------")

# Modelo B
accuracy_b = accuracy_score(y_realidad, y_pred_b)
precision_b = precision_score(y_realidad, y_pred_b)
recall_b = recall_score(y_realidad, y_pred_b)
confusion_matrix_b = confusion_matrix(y_realidad, y_pred_b)

print ("Modelo B")
print ("Accuracy:", accuracy_b)
print ("Precision:", precision_b)
print ("Recall:", recall_b)
print ("Confusion Matrix:\n", confusion_matrix_b)
```

* Me da esto

```
Modelo A
Accuracy: 0.8636363636363636
Precision: 1.0
Recall: 0.25
Confusion Matrix:
 [[18  0]
 [ 3  1]]
------------------------------
Modelo B
Accuracy: 0.8181818181818182
Precision: 0.5
Recall: 0.75
Confusion Matrix:
 [[15  3]
 [ 1  3]]
```

* Precision, recall y accuracy dan valores entre 0 y 1
* Si me baso en el accuracy me quedo con el Modelo a -> 
* Precision:
  * Nos da una confianza alta en lo que dice el modelo que es fraude
  * Modelo a :  1 -> Todo lo que marco como fraude lo era.
  * Modelo b :  0.5 ->  La mitad de lo que detecto como fraude no lo era (o era fraude)
* Recall
  * Nos dice que dejo pasar por alto el modelo
  * En los casos donde es caro que se te pase por alto un positivo (fraude) buscas recall alto
  * Modelo B: 0.25 -> Detecto una cuarta parte de los fraudes que hay
  * Modelo B : 0.75 -> De los fraudes detecto el 75% (1 lo dejo pasar)

# Pandas

* Estructuras
  * Series        ->     Vector/Array
  * Dataframe     ->     Tabla
 
* Permite levantar datos desde un excel o un archivo CSV

## Declaramos un Data frame

```python
import pandas as pd

ventas = pd.DataFrame({
    "producto" : ["Notebook", "Mouse", "Teclado", "Monitor", "Impresora", "WebCam"],
    "precio" : [1000, 50, 80, 300, 150, 70],
    "stock" : [10, 100, 50, 20, 15, 30]
})
```

## Metodos de Exploracion de Datos

* Los metodos son
 * head
 * tail
 * info
 * describe

* Ver los primeros
```python
ventas.head(3)
```

* Ver los ultimo
```python
ventas.tail(3)
```

* Ver todos
```python
print(ventas)
```

* Info de todo
  
```python
ventas.info()
```

* Descripcion de metricas basicas de toda la tabla (DataFrame)

```python
ventas.describe()
```

* Si queremos metricas de una columna sola

```python
ventas["precio"].describe()
```

* Cuando es una sola columna te devuelve un objeto de tipo Series

```python
print(type(ventas["precio"]))
```

* Para ver columnas y tipos

```python
print(ventas.columns)
print(ventas.dtypes)
```

> [!NOTE]
> Para entrenar modelos siempre vamos a usar columnas numericas (int o float)

## Seleccionar Datos

```

# Select precio from ventas
precios = ventas["precio"]
print(precios)

# Select producto,stock from ventas
# No cofundirse con el [[]] es un tema de python, si se pone un [] se obtiene una serie, si se pone [[]] se obtiene un dataframe
productos_stock = ventas[ ["producto", "stock"] ]
print(productos_stock)
```

## Filtrar datos

```python
# select * from ventas where precio > 200

ventas_mayores_200 = ventas[ ventas["precio"] > 200 ]

print(ventas_mayores_200)

print("-------------")

# sellect prodcuto from vetnas where precio > 200
productos_mayores_200 = ventas[ventas["precio"] > 200]["producto"]
print(productos_mayores_200)
```

# Explorar Datos

* Metodos
 * unique
 * value_counts 

* Chequeo los datos
  * Ver si cada columna categorica tiene la cantidad de posibilidades esperadas (a veces con la mayuscula, minuscula, trae problemas)
  * Como se distribuyen los valores posibles (con esto hago un histograma)

```python
personas = pd.DataFrame({
    "nombre" : ["Juan", "Pedro", "Maria", "Ana", "Luis"],
    "sexo" : ["M", "M", "F", "F", "M"],
    "altura" : [1.70, 1.80, 1.60, 1.65, 1.75]
})

#Laveriable sexo cuantos datos posibles esperari que tuviera? -> 2
print(personas["sexo"].unique())
print()
print(personas["sexo"].value_counts())
```

* Un ejemplo donde hay datos mal cargados y tengo que modificar columnas existentes

```python
# Modificar columnas

# Tengo un dataset con valores que estan mal cargados 
personas = pd.DataFrame({
    "nombre" : ["Juan", "Pedro", "Maria", "Ana", "Luis"],
    "sexo" : ["M", "m", "f", "F", "M"],
    "altura" : [1.70, 1.80, 1.60, 1.65, 1.75]
})

#Laveriable sexo cuantos datos posibles esperari que tuviera? -> 2
print(personas["sexo"].unique())
print()
print(personas["sexo"].value_counts())


print("-------------")

# Voy a unificar los valores de la columna sexo, para que todos sean mayusculas
personas["sexo"] = personas["sexo"].str.upper()  #<<< EStoy modificando una columna existente
print()
print(personas["sexo"].unique())
print()
print(personas["sexo"].value_counts())
```

* Creacion de columnas nuevas y eliminacion de columnas

```python
#Creacion de columnas nuevas
personas["altura_cm"] = personas["altura"] * 100  #<<< Estoy creando una columna nueva
print(personas)

#Ahora las variables altura y altura_cm son colineales / estan correlacionadas, voy a borrar una
personas.drop(columns=["altura"], inplace=True)  #<<< Estoy borrando una columna existente
```
