# Guiaa - Evaluación de Modelos de Clasificación con la Matriz de Confusión

## Objetivos

Al finalizar esta guía podrás:

* Comprender cómo evaluar modelos de clasificación.
* Interpretar una matriz de confusión.
* Comparar distintos algoritmos de Machine Learning.
* Entender por qué existen métricas más avanzadas como Sensibilidad, Especificidad y ROC AUC.

---

# 1. El problema

Supongamos que queremos construir un modelo capaz de detectar enfermedades cardíacas.

Cada paciente posee información como:

* Edad
* Sexo
* Presión arterial
* Colesterol
* Frecuencia cardíaca
* Nivel de azúcar
* Dolor de pecho
* ...

Nuestro objetivo será predecir una de dos clases:

* ❤️ Enfermo
* ✅ Sano

Este es un problema de **clasificación binaria**.

---

# 2. ¿Qué algoritmos podríamos utilizar?

Existen muchos algoritmos capaces de resolver este problema.

Por ejemplo:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Random Forest

Cada uno utiliza una estrategia diferente para realizar sus predicciones.

La gran pregunta es...

> **¿Cómo sabemos cuál funciona mejor?**

---

# 3. Entrenamiento y evaluación

No podemos evaluar un modelo utilizando los mismos datos con los que aprendió.

Por eso dividimos el dataset en dos partes:

* Datos de entrenamiento
* Datos de prueba

```
Dataset
│
├── Train (80%)
└── Test (20%)
```

El modelo aprende únicamente con el conjunto de entrenamiento.

Luego realiza predicciones sobre datos que nunca había visto.

> En proyectos reales suele utilizarse además **Cross Validation**, que permite obtener una evaluación más robusta utilizando diferentes particiones del dataset.

---

# 4. Entrenamos varios modelos

Entrenaremos tres algoritmos:

* Logistic Regression
* KNN
* Random Forest

Todos utilizarán exactamente el mismo conjunto de entrenamiento y serán evaluados sobre el mismo conjunto de prueba.

---

# 5. ¿Cómo evaluamos un modelo?

Supongamos que Random Forest realizó una predicción para todos los pacientes del conjunto de prueba.

Ahora debemos comparar:

* lo que el modelo predijo
* contra la realidad

La herramienta más utilizada para hacerlo es la **Matriz de Confusión**.

---

# 6. La Matriz de Confusión

Una matriz de confusión resume todas las predicciones realizadas por un modelo.

Las filas representan la **clase real**.

Las columnas representan la **clase predicha**.

|                      |        Predijo Sano |     Predijo Enfermo |
| -------------------- | ------------------: | ------------------: |
| **Paciente Sano**    |  True Negative (TN) | False Positive (FP) |
| **Paciente Enfermo** | False Negative (FN) |  True Positive (TP) |

Cada una de estas cuatro celdas tiene un significado diferente.

### True Positive (TP)

El paciente estaba enfermo y el modelo lo detectó correctamente.

### True Negative (TN)

El paciente estaba sano y el modelo acertó.

### False Positive (FP)

El paciente estaba sano pero el modelo dijo que estaba enfermo.

También se conoce como **falso positivo**.

### False Negative (FN)

El paciente estaba enfermo pero el modelo indicó que estaba sano.

Este suele ser el error más grave en aplicaciones médicas.

---

# 7. Ejemplo: Random Forest

Supongamos que obtenemos la siguiente matriz.

|                      | Predijo Sano | Predijo Enfermo |
| -------------------- | -----------: | --------------: |
| **Paciente Sano**    |      **142** |          **22** |
| **Paciente Enfermo** |       **29** |         **110** |

Esto significa que:

* Detectó correctamente 110 pacientes enfermos.
* Detectó correctamente 142 pacientes sanos.
* Marcó incorrectamente como enfermos a 22 pacientes sanos.
* No detectó 29 pacientes que realmente estaban enfermos.

---

# 8. Ejemplo: KNN

Ahora observemos otro algoritmo.

|                      | Predijo Sano | Predijo Enfermo |
| -------------------- | -----------: | --------------: |
| **Paciente Sano**    |      **107** |          **53** |
| **Paciente Enfermo** |       **64** |          **79** |

A simple vista observamos que comete muchos más errores.

---

# 9. Comparando ambos modelos

Random Forest

|                      | Predijo Sano | Predijo Enfermo |
| -------------------- | -----------: | --------------: |
| **Paciente Sano**    |          142 |              22 |
| **Paciente Enfermo** |           29 |             110 |

KNN

|                      | Predijo Sano | Predijo Enfermo |
| -------------------- | -----------: | --------------: |
| **Paciente Sano**    |          107 |              53 |
| **Paciente Enfermo** |           64 |              79 |

Random Forest presenta:

* Más aciertos.
* Menos falsos positivos.
* Menos falsos negativos.

Por lo tanto, parece ser una mejor opción para este problema.

---

# 10. Comparando con Logistic Regression

Supongamos ahora la siguiente matriz.

|                      | Predijo Sano | Predijo Enfermo |
| -------------------- | -----------: | --------------: |
| **Paciente Sano**    |      **139** |          **22** |
| **Paciente Enfermo** |       **32** |         **112** |

Observamos que sus resultados son muy similares a Random Forest.

En este caso ya no resulta tan sencillo decidir cuál es mejor únicamente observando la matriz.

---

# 11. ¿Cuál debería elegir?

La respuesta depende del problema.

En algunos casos puede interesar:

* Detectar la mayor cantidad posible de pacientes enfermos.
* Reducir la cantidad de falsas alarmas.
* Maximizar la cantidad total de aciertos.

No existe un algoritmo que siempre sea el mejor.

La elección dependerá del objetivo del negocio o del problema que se intenta resolver.

---

# 12. ¿La matriz de confusión alcanza?

La matriz de confusión es una excelente herramienta para comprender el comportamiento de un modelo.

Sin embargo, cuando queremos comparar modelos de forma más objetiva solemos calcular métricas derivadas de ella, por ejemplo:

* Accuracy
* Precision
* Recall (Sensibilidad)
* Especificidad (Specificity)
* F1 Score
* ROC AUC

Estas métricas permiten resumir el rendimiento del modelo mediante uno o más valores numéricos y serán el siguiente paso en nuestro estudio.


Pedagógicamente, esa progresión suele funcionar muy bien porque cada concepto surge para resolver una limitación del anterior, en lugar de presentarse como una lista de métricas.
