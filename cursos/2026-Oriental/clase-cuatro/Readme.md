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


# Pandas
