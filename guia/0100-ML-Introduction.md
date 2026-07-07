# Clase: Introducción Suave al Machine Learning

## Objetivos

Al finalizar esta clase los estudiantes serán capaces de:

* Comprender qué es Machine Learning.
* Diferenciar clasificación y regresión.
* Entender el propósito de los datos de entrenamiento y prueba.
* Comprender por qué un modelo muy complejo no necesariamente es mejor.
* Entender la idea de generalización.
* Comprender la importancia de evaluar modelos con datos nuevos.

## Secuencia

```
Árbol de decisión
        ↓
Clasificación
        ↓
Regresión (línea negra)
        ↓
Training Data
        ↓
Línea negra vs curva verde
        ↓
Testing Data
        ↓
Calcular errores
        ↓
Comparar errores
        ↓
Generalización
        ↓
Volver al árbol
        ↓
Conclusión
```


***

# ¿Qué es Machine Learning?

Cuando la gente escucha "Machine Learning" suele pensar en:

* Inteligencia Artificial
* Redes neuronales
* Algoritmos complejos
* Grandes volúmenes de datos

Pero la idea fundamental es mucho más simple.

> Machine Learning consiste en utilizar ejemplos para construir modelos capaces de realizar predicciones o clasificaciones sobre datos nuevos.

Toda la clase de hoy gira alrededor de esta definición.

***

# Parte 1 — Un Árbol de Decisión

Supongamos que queremos predecir si una persona disfrutará un curso de Machine Learning.

Podemos hacer algunas preguntas.

## Mostrar árbol

```text
¿Te gusta programar?

        Sí
         │
         ▼

¿Te gusta resolver problemas?

     Sí                No
      │                 │
      ▼                 ▼

Le gustará        No le gustará
el curso          el curso
```

***

## ¿Qué está haciendo el árbol?

El árbol observa respuestas y termina realizando una predicción.

La salida solamente puede ser:

* Sí
* No

No existen valores intermedios.

***

# Clasificación

Cuando un algoritmo asigna una categoría estamos ante un problema de:

# Clasificación

Ejemplos:

* Spam / No spam
* Fraude / No fraude
* Enfermo / Sano
* Aprobado / Suspendido
* Le gusta el curso / No le gusta

***

## Primera idea importante

Este árbol es una forma de Machine Learning.

Porque utiliza ejemplos para construir reglas que permiten clasificar nuevos casos.

***

# Parte 2 — Predicción Numérica (Regresión)

Veamos ahora otro problema.

Supongamos que disponemos de información de alumnos anteriores.

Para cada alumno conocemos:

* Horas de estudio
* Nota obtenida

***

## Mostrar gráfico

```text
Nota

10 |                         ●
 9 |
 8 |                    ●
 7 |
 6 |              ●
 5 |
 4 |        ●
 3 |
 2 |  ●

    +--------------------------
      1   2   3   4   5   6

       Horas de estudio
```

Cada punto representa un alumno.

***

## ¿Qué observamos?

A medida que aumentan las horas de estudio:

→ La nota tiende a aumentar.

Existe una tendencia.

***

## Ajustando una línea

Podemos representar esa tendencia mediante una línea.

```text
Nota

10 |                         ●
 9 |                       /
 8 |                    ●/
 7 |                  /
 6 |              ● /
 5 |            /
 4 |        ● /
 3 |      /
 2 |  ● /

    +--------------------------
      1   2   3   4   5   6

       Horas de estudio
```

***

## Realizando una predicción

Aparece un alumno nuevo.

Sabemos que estudió 4 horas.

Queremos estimar qué nota podría obtener.

```text
Nota

10 |
 9 |
 8 |
 7 |
 6 |            ★
 5 |
 4 |
 3 |

    +--------------------------
      1   2   3   4   5   6
```

Utilizamos la línea para realizar una predicción.

Por ejemplo:

> El modelo predice una nota cercana a 6,5.

***

# Regresión

En este caso la salida no es una categoría.

La salida es un número.

Por eso estamos ante un problema de:

# Regresión

***

# Resumiendo

Hasta ahora hemos visto dos tipos de problemas.

| Problema      | Resultado         |
| ------------- | ----------------- |
| Clasificación | Categorías        |
| Regresión     | Valores numéricos |

***

## Idea central

Gran parte del Machine Learning consiste en resolver uno de estos dos problemas:

* Clasificar
* Predecir valores

***

# Parte 3 — Datos de Entrenamiento

Los ejemplos que usamos para construir un modelo reciben un nombre.

# Training Data

o

# Datos de Entrenamiento

***

En nuestro ejemplo:

Los alumnos utilizados para dibujar la línea son Training Data.

```text
● ● ● ● ●
```

***

## ¿Qué hace el modelo?

Observa esos ejemplos.

Y aprende patrones.

***

# Parte 4 — Dos Modelos Distintos

Supongamos ahora que entrenamos dos modelos diferentes.

***

## Modelo A

Una línea simple.

```text
Nota

10 |                    ●
 9 |
 8 |                ●
 7 |
 6 |            ●
 5 |
 4 |        ●
 3 |
 2 |    ●

     -----------------
```

***

## Modelo B

Una curva muy compleja.

```text
Nota

10 |                 ~~●
 9 |           ~~~~~~
 8 |        ●~
 7 |~~~~~~~
 6 |      ●~
 5 |
 4 | ●~~~
```

***

## Pregunta para la clase

¿Cuál parece mejor?

La mayoría responderá:

> La curva.

Porque pasa exactamente por todos los puntos.

***

# Pero...

Recordemos algo.

El objetivo del Machine Learning no es explicar perfectamente los datos antiguos.

El objetivo es:

> Realizar predicciones sobre datos nuevos.

***

# Parte 5 — Datos de Prueba

Necesitamos nuevos alumnos.

Alumnos que no participaron del entrenamiento.

Estos datos reciben el nombre de:

# Testing Data

o

# Datos de Prueba

***

## Resumen

Training Data

```text
● ● ● ● ●
```

Sirve para aprender.

***

Testing Data

```text
★ ★ ★ ★
```

Sirve para evaluar.

***

# Parte 6 — Probando la Línea

Alumno 1

```text
Nota real = 8
Predicción = 7

Error = 1
```

***

Alumno 2

```text
Nota real = 6
Predicción = 7

Error = 1
```

***

Alumno 3

```text
Nota real = 9
Predicción = 8

Error = 1
```

***

Alumno 4

```text
Nota real = 7
Predicción = 7

Error = 0
```

***

## Error total

```text
1 + 1 + 1 + 0 = 3
```

***

# Parte 7 — Probando la Curva

Alumno 1

```text
Nota real = 8
Predicción = 6

Error = 2
```

***

Alumno 2

```text
Nota real = 6
Predicción = 9

Error = 3
```

***

Alumno 3

```text
Nota real = 9
Predicción = 7

Error = 2
```

***

Alumno 4

```text
Nota real = 7
Predicción = 10

Error = 3
```

***

## Error total

```text
2 + 3 + 2 + 3 = 10
```

***

# Comparación Final

```text
Modelo A (línea)  = Error 3

Modelo B (curva)  = Error 10
```

***

# ¿Qué ocurrió?

La curva aprendió perfectamente los datos de entrenamiento.

Pero se equivocó más con datos nuevos.

La línea simple aprendió mejor la tendencia general.

Por eso hizo mejores predicciones.

***

# Enseñanza Fundamental

No gana el modelo que mejor funciona con Training Data.

Gana el modelo que mejor funciona con Testing Data.

Esta es una de las ideas más importantes de todo Machine Learning.

***

# Overfitting

Cuando un modelo aprende demasiado los datos de entrenamiento:

* Memoriza ejemplos
* Aprende ruido
* Pierde capacidad de generalización

Decimos que existe:

# Overfitting

***

## Idea intuitiva

```text
Memorizar ≠ Aprender
```

Un modelo útil debe generalizar.

***

# Parte 8 — Volviendo al Árbol de Decisión

Recordemos el árbol del principio.

Lo construimos usando Training Data.

Ahora vamos a probarlo utilizando Testing Data.

***

## Ejemplo

Alumno:

* Le gusta programar
* Le gusta resolver problemas

El árbol predice:

```text
Le gustará el curso
```

Y efectivamente:

```text
Le gustó el curso
```

Correcto.

***

## Otro ejemplo

Alumno:

* No le gusta programar
* No le gusta resolver problemas

El árbol predice:

```text
No le gustará el curso
```

Pero en la realidad:

```text
Sí le gustó el curso
```

Error.

***

## Evaluación

Repetimos este proceso para todas las personas del conjunto de Testing.

Así sabemos qué tan bueno es realmente el árbol.

***

# ¿Importa Qué Algoritmo Usemos?

Actualmente existen muchísimos algoritmos.

* KNN
* Árboles de decisión
* Regresión logística
* Random Forest
* SVM
* Gradient Boosting
* Redes neuronales

Todos son distintos.

Pero todos se evalúan de la misma manera.

***

# Doble BAM

Lo importante no es qué tan sofisticado sea el algoritmo.

Lo importante es:

> Qué tan bien funciona sobre datos que nunca vio.

***

# Resumen Final

Machine Learning consiste en construir modelos utilizando ejemplos.

Los modelos suelen resolver dos grandes tipos de problemas:

* Clasificación
* Regresión

Los modelos aprenden usando:

* Training Data

Los modelos se evalúan usando:

* Testing Data

Un modelo puede ajustarse perfectamente a los datos de entrenamiento y aun así ser malo prediciendo datos nuevos.

El objetivo final no es memorizar.

El objetivo final es generalizar.

***

## Mensaje de cierre

> Machine Learning no trata de encontrar el modelo más complejo.
>
> Trata de encontrar el modelo que mejor predice datos que todavía no conocemos. 🚀

Esta versión mantiene prácticamente la misma secuencia y enseñanza del vídeo de Josh Starmer, sustituyendo únicamente el ejemplo de StatQuest por ejemplos relacionados con estudiantes y cursos.
