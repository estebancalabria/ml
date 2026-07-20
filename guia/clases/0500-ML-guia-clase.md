# Guía de Clase 5 - Normalización de Variables, Algoritmos Basados en Distancia y Árboles

## Objetivo

Al finalizar la clase los alumnos deberían:

* Comprender cuándo un algoritmo necesita normalizar los datos.
* Diferenciar algoritmos basados en distancia de algoritmos basados en árboles.
* Conocer los principales algoritmos de Machine Learning para clasificación y regresión.
* Aplicar distintas técnicas de escalado utilizando Scikit-Learn.
* Comparar StandardScaler, MinMaxScaler y RobustScaler.
* Entrenar distintos modelos sobre un mismo problema y comparar resultados.
* Comprender cuándo conviene utilizar cada familia de algoritmos.
* Introducir el concepto de aprendizaje no supervisado y clustering.

---

# Agenda

1. Repaso de la clase anterior
2. Algoritmos basados en distancia vs algoritmos basados en árboles
3. Principales algoritmos de Machine Learning
4. Laboratorio 630 - Escalado de variables numéricas
5. **BREAK**
6. Kahoot
7. Ejemplo California Housing (Laboratorio 755)
8. **BREAK**
9. Kahoot
10. Ejemplo Titanic (Laboratorio 710)
11. Introducción al aprendizaje no supervisado y Clustering (si queda tiempo)
12. Próximos pasos

---

# 1. Repaso

Realizar un repaso breve de la clase anterior.

Recordar:

* Exploración de datos
* Variables categóricas y numéricas
* Label Encoding
* Ordinal Encoding
* One-Hot Encoding
* Árboles de decisión
* Interpretación del árbol generado

No repetir toda la teoría. Solamente recuperar los conceptos necesarios para la clase actual.

---

# 2. Algoritmos basados en distancia vs algoritmos basados en árboles

Antes de comenzar con el escalado es importante entender que **no todos los algoritmos utilizan la información de la misma forma**.

Existen muchas familias de algoritmos, pero hoy nos enfocaremos en dos.

## Algoritmos basados en distancia

Estos algoritmos calculan qué tan parecidos son dos registros.

Utilizan alguna métrica de distancia (generalmente distancia euclídea).

Si una variable tiene valores mucho más grandes que otra, dominará completamente el cálculo.

Por ese motivo necesitan que todas las variables estén en escalas similares.

Ejemplos:

* K-Nearest Neighbors (KNN)
* K-Means
* Support Vector Machines (SVM)
* Redes Neuronales (recomendado)
* PCA

Generalmente requieren:

* StandardScaler
* MinMaxScaler
* RobustScaler

---

## Algoritmos basados en árboles

Los árboles no calculan distancias.

Construyen reglas del tipo:

```
¿Edad > 30?

¿Ingreso < 50000?

¿Antigüedad > 5 años?
```

Como solamente comparan valores, la escala no modifica el resultado.

Por ese motivo **no necesitan normalización**.

Ejemplos:

* Decision Tree
* Random Forest
* Extra Trees
* Gradient Boosting
* XGBoost
* LightGBM
* CatBoost

---

## Resumen

| Basados en distancia  | Basados en árboles     |
| --------------------- | ---------------------- |
| Calculan distancias   | Construyen reglas      |
| Necesitan escalado    | No necesitan escalado  |
| Sensibles a la escala | Poco sensibles         |
| Ejemplo: KNN          | Ejemplo: Random Forest |

---

# 3. Principales algoritmos de Machine Learning

## Clasificación

### Basados en distancia

* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)

### Basados en árboles

* Decision Tree
* Random Forest
* Extra Trees
* Gradient Boosting
* XGBoost
* LightGBM
* CatBoost

### Otros muy utilizados

* Logistic Regression
* Naive Bayes
* Redes Neuronales

---

## Regresión

### Basados en distancia

* KNN Regressor
* Support Vector Regression (SVR)

### Basados en árboles

* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Regressor
* XGBoost Regressor
* LightGBM Regressor
* CatBoost Regressor

### Otros muy utilizados

* Linear Regression
* Ridge
* Lasso
* Elastic Net
* Redes Neuronales

---

## Aprendizaje no supervisado

### Clustering

* K-Means
* DBSCAN
* Agglomerative Clustering
* Mean Shift

### Reducción de dimensionalidad

* PCA
* t-SNE
* UMAP

Aclarar que estos algoritmos trabajan sin una variable objetivo.

---

# 4. Laboratorio 630 - Escalado de Variables Numéricas

Realizar el laboratorio completo.

Objetivos:

* Comprender por qué algunas variables deben normalizarse.
* Comparar StandardScaler.
* Comparar MinMaxScaler.
* Comparar RobustScaler.
* Analizar el efecto de los outliers.
* Entender cuál utilizar según el problema.

Durante el laboratorio discutir:

* ¿Qué ocurre cuando una variable tiene valores mucho mayores que otra?
* ¿Por qué KNN cambia tanto después del escalado?
* ¿Por qué RobustScaler funciona mejor cuando existen outliers?

---

# BREAK

10 a 15 minutos.

---

# Kahoot

Repasar:

* Distancia
* Árboles
* StandardScaler
* MinMaxScaler
* RobustScaler
* Outliers

Resolver dudas antes de continuar.

---

# 5. Laboratorio 755 - California Housing

Aplicar distintos algoritmos de regresión sobre el mismo dataset.

Comparar:

* Linear Regression
* KNN Regressor
* Decision Tree Regressor
* Random Forest Regressor

Analizar:

* Error obtenido.
* Influencia del escalado.
* Comparación entre modelos.

Preguntas para discutir:

* ¿Cuál obtuvo mejor desempeño?
* ¿Cuál fue más rápido?
* ¿Qué ocurrió al escalar los datos?
* ¿Todos mejoraron?

---

# BREAK

10 minutos.

---

# Kahoot

Repaso rápido de:

* Regresión
* RMSE
* MAE
* Escalado
* KNN
* Random Forest

---

# 6. Laboratorio 710 - Clasificación con Titanic

Resolver el laboratorio completo.

Comparar distintos clasificadores sobre el mismo dataset.

Ejemplos:

* Logistic Regression
* KNN
* Decision Tree
* Random Forest

Comparar:

* Accuracy
* Precision
* Recall
* F1 Score

Discutir:

* ¿Qué algoritmo obtuvo mejores resultados?
* ¿Qué diferencias existen entre un modelo basado en distancia y uno basado en árboles?
* ¿Qué algoritmo elegiríamos para producción?

---

# 7. Introducción al aprendizaje no supervisado (si queda tiempo)

Presentar el concepto de aprendizaje no supervisado.

Explicar que en estos problemas **no existe una columna objetivo**.

Objetivos habituales:

* Descubrir grupos naturales.
* Detectar anomalías.
* Reducir dimensionalidad.
* Encontrar patrones ocultos.

Introducir K-Means como primer algoritmo de clustering.

Mostrar un ejemplo sencillo con datos bidimensionales y explicar cómo el algoritmo agrupa observaciones similares.

No profundizar todavía en la implementación; el objetivo es dejar preparado el terreno para la próxima clase.

---

# Próximos pasos

...


Es una secuencia pedagógica donde cada bloque justifica al siguiente, evitando que la normalización aparezca como un concepto aislado.
