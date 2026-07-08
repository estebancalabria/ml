# Clase: Matriz de Confusión (Confusion Matrix)

## Objetivos

Al finalizar esta clase los estudiantes serán capaces de:

* Comprender por qué es necesario evaluar modelos de clasificación.
* Entender cómo se comparan distintos algoritmos de Machine Learning.
* Comprender la estructura de una matriz de confusión.
* Interpretar correctamente filas y columnas de una matriz de confusión.
* Identificar True Positives, True Negatives, False Positives y False Negatives.
* Comprender el significado de los aciertos y errores de clasificación.
* Utilizar matrices de confusión para comparar modelos.
* Comprender matrices binarias y multiclase.
* Entender que la matriz de confusión es la base de métricas más avanzadas.

## Secuencia

```text
Tenemos varios algoritmos
        ↓
¿Cómo elegir el mejor?
        ↓
Training Data
        ↓
Testing Data
        ↓
Realizar predicciones
        ↓
Aciertos y errores
        ↓
Matriz de Confusión
        ↓
True Positive
True Negative
False Positive
False Negative
        ↓
Interpretación de la diagonal
        ↓
Comparación de algoritmos
        ↓
Limitaciones
        ↓
Matrices multiclase
        ↓
Matrices NxN
        ↓
Precision, Recall y ROC
        ↓
Conclusión
```

***

# Introducción

En Machine Learning existen muchos algoritmos distintos para resolver problemas de clasificación.

Por ejemplo:

* K Nearest Neighbors (KNN)
* Árboles de decisión
* Random Forest
* Regresión Logística
* Support Vector Machines (SVM)
* Gradient Boosting

Todos ellos pueden utilizarse para hacer predicciones.

La pregunta es:

> ¿Cómo sabemos cuál funciona mejor?

***

# Un Problema de Clasificación

Imaginemos que queremos predecir si una persona desarrollará una enfermedad cardíaca.

Disponemos de datos como:

* Dolor de pecho
* Circulación sanguínea
* Arterias bloqueadas
* Peso
* Otros indicadores clínicos

Y queremos predecir:

```text
Tiene enfermedad cardíaca
```

o

```text
No tiene enfermedad cardíaca
```

***

# Tenemos Varios Algoritmos

Podemos entrenar distintos modelos.

Por ejemplo:

* KNN
* Random Forest
* Regresión Logística

Todos utilizan exactamente los mismos datos.

Pero no necesariamente obtienen los mismos resultados.

***

# ¿Cómo Comparamos Modelos?

No podemos simplemente decir:

```text
Este algoritmo parece bueno.
```

Necesitamos una forma objetiva de medir su rendimiento.

Para eso utilizamos datos que el algoritmo nunca vio durante el entrenamiento.

***

# Training Data y Testing Data

Dividimos los datos en dos grupos.

## Training Data

Se utiliza para entrenar el modelo.

***

## Testing Data

Se utiliza para evaluar el modelo.

***

```text
Datos originales

       ▼

+-------------------+
|  Training Data    |
+-------------------+

+-------------------+
|   Testing Data    |
+-------------------+
```

***

# Entrenando los Modelos

Cada algoritmo aprende utilizando el conjunto de entrenamiento.

Por ejemplo:

```text
Training Data

        ↓

Random Forest

        ↓

Modelo entrenado
```

***

```text
Training Data

        ↓

KNN

        ↓

Modelo entrenado
```

***

```text
Training Data

        ↓

Regresión Logística

        ↓

Modelo entrenado
```

***

# Probando los Modelos

Ahora utilizamos los datos de prueba.

El modelo debe hacer predicciones sobre pacientes que nunca vio antes.

***

### Paciente 1

```text
Realidad: Tiene enfermedad cardíaca

Predicción: Tiene enfermedad cardíaca
```

Correcto.

***

### Paciente 2

```text
Realidad: Tiene enfermedad cardíaca

Predicción: No tiene enfermedad cardíaca
```

Incorrecto.

***

### Paciente 3

```text
Realidad: No tiene enfermedad cardíaca

Predicción: Tiene enfermedad cardíaca
```

Incorrecto.

***

### Paciente 4

```text
Realidad: No tiene enfermedad cardíaca

Predicción: No tiene enfermedad cardíaca
```

Correcto.

***

# Necesitamos Resumir los Resultados

Cuando tenemos cientos o miles de pacientes necesitamos una herramienta que nos permita responder rápidamente:

* ¿Cuántos casos acertó?
* ¿Cuántos casos falló?
* ¿Qué tipo de errores cometió?

Para eso utilizamos la:

# Matriz de Confusión

***

# Estructura Básica

Las filas representan:

```text
Lo que predijo el algoritmo
```

Las columnas representan:

```text
La realidad
```

***

```text
                       REAL

                 Sí          No

PREDICE Sí       ?           ?

PREDICE No       ?           ?
```

***

# True Positive (TP)

Son pacientes que tenían la enfermedad y fueron correctamente identificados.

```text
                       REAL

                 Sí          No

PREDICE Sí      TP           ?

PREDICE No       ?           ?
```

***

# True Negative (TN)

Son pacientes que no tenían la enfermedad y fueron correctamente identificados.

```text
                       REAL

                 Sí          No

PREDICE Sí      TP           ?

PREDICE No       ?          TN
```

***

# False Negative (FN)

Son pacientes que tenían la enfermedad pero fueron clasificados como sanos.

```text
                       REAL

                 Sí          No

PREDICE Sí      TP           ?

PREDICE No      FN          TN
```

***

# False Positive (FP)

Son pacientes sanos que fueron clasificados como enfermos.

```text
                       REAL

                 Sí          No

PREDICE Sí      TP          FP

PREDICE No      FN          TN
```

***

# Matriz Completa

```text
                       REAL

                 Sí          No

PREDICE Sí      TP          FP

PREDICE No      FN          TN
```

***

# Ejemplo Real

Supongamos que obtenemos los siguientes resultados.

```text
                       REAL

                Enfermo     Sano

PREDICE Enfermo   142         22

PREDICE Sano       29        110
```

***

# Interpretación

```text
142
```

Pacientes enfermos correctamente identificados.

***

```text
110
```

Pacientes sanos correctamente identificados.

***

```text
29
```

Pacientes enfermos clasificados incorrectamente como sanos.

***

```text
22
```

Pacientes sanos clasificados incorrectamente como enfermos.

***

# La Diagonal

Observemos la diagonal principal.

```text
142       -

 -       110
```

Estos son los aciertos del algoritmo.

***

Los valores fuera de la diagonal representan errores.

```text
 -       22

29        -
```

Estos son los fallos del algoritmo.

***

# Regla Fundamental

## Diagonal = Correcto

## Fuera de la diagonal = Error

***

# Comparando Algoritmos

Ahora comparemos dos modelos.

***

## Random Forest

```text
                       REAL

                Enfermo     Sano

PREDICE Enfermo   142        22

PREDICE Sano       29       110
```

***

## K Nearest Neighbors

```text
                       REAL

                Enfermo     Sano

PREDICE Enfermo   107        45

PREDICE Sano       64        79
```

***

# ¿Cuál Parece Mejor?

Observemos la diagonal.

```text
Random Forest

142 + 110 = 252
```

***

```text
KNN

107 + 79 = 186
```

***

El Random Forest acertó más casos.

Por lo tanto parece ser una mejor alternativa para este problema.

***

# ¿Y Si Dos Modelos Son Muy Parecidos?

Puede ocurrir que dos algoritmos produzcan matrices muy similares.

Por ejemplo:

* Random Forest
* Regresión Logística

En ese caso resulta difícil decidir cuál es mejor observando solamente la matriz de confusión.

***

# Limitaciones

La matriz de confusión es muy útil.

Pero no siempre es suficiente.

Por eso existen métricas derivadas como:

* Precision
* Recall
* Sensitivity
* Specificity
* F1 Score
* ROC Curve
* ROC AUC

Las estudiaremos más adelante.

***

# Matrices Multiclase

Hasta ahora sólo hemos trabajado con dos posibles resultados.

```text
Enfermo
o
Sano
```

Esto produce una matriz:

```text
2 x 2
```

***

Pero muchos problemas tienen más de dos categorías.

***

# Ejemplo: Película Favorita

Supongamos que queremos predecir la película favorita de una persona.

Las opciones son:

* Película A
* Película B
* Película C

***

Ahora la matriz ya no será 2x2.

Será:

```text
3 x 3
```

***

# Matriz Multiclase

```text
                 REAL

             A    B    C

PREDICE A    ?    ?    ?

PREDICE B    ?    ?    ?

PREDICE C    ?    ?    ?
```

***

# La Regla Sigue Siendo la Misma

La diagonal contiene los aciertos.

```text
A → A

B → B

C → C
```

***

Todo lo que esté fuera de la diagonal representa errores.

```text
A → B

A → C

B → A

B → C

C → A

C → B
```

***

# Interpretación Visual

```text
✓  x  x

x  ✓  x

x  x  ✓
```

La diagonal indica las clasificaciones correctas.

***

# Generalización

Dos clases:

```text
2 x 2
```

***

Tres clases:

```text
3 x 3
```

***

Cuatro clases:

```text
4 x 4
```

***

Diez clases:

```text
10 x 10
```

***

Cuarenta clases:

```text
40 x 40
```

***

# Regla General

Si nuestro problema tiene:

```text
N categorías
```

La matriz tendrá:

```text
N filas

N columnas
```

***

# Idea Fundamental

La matriz de confusión no está limitada a problemas binarios.

Puede utilizarse para cualquier problema de clasificación.

***

# Resumen Final

La matriz de confusión es una herramienta fundamental para evaluar clasificadores.

Permite observar:

* Aciertos
* Errores
* Tipo de errores

Las filas representan:

* Lo que predice el modelo

Las columnas representan:

* La realidad

La diagonal contiene:

* Clasificaciones correctas

Fuera de la diagonal encontramos:

* Clasificaciones incorrectas

Las matrices pueden ser:

* 2x2
* 3x3
* 4x4
* NxN

Además, constituyen la base para métricas más avanzadas como:

* Precision
* Recall
* Sensitivity
* Specificity
* F1 Score
* ROC Curve
* ROC AUC

***

## Mensaje de cierre

> Una matriz de confusión nos muestra exactamente dónde acierta y dónde se equivoca un modelo de Machine Learning.
>
> Es el punto de partida para entender si un clasificador realmente funciona. 🚀
