# Clase: Introducción al Machine Learning

## Objetivos

Al finalizar esta clase los estudiantes deberían ser capaces de:

- Comprender qué es Machine Learning.
- Diferenciar clasificación y regresión.
- Entender qué son los datos de entrenamiento y prueba.
- Comprender por qué un modelo que aprende "demasiado" no siempre es el mejor.
- Entender que el objetivo del Machine Learning es generalizar y no memorizar.

---

# ¿Qué es Machine Learning?

Una definición muy simple sería:

> Machine Learning consiste en entrenar un algoritmo utilizando ejemplos para que pueda realizar predicciones o clasificaciones sobre datos nuevos.

Es decir, en lugar de programar reglas manualmente, dejamos que el algoritmo descubra los patrones presentes en los datos.

---

# Ejemplo 1: Clasificación

Vamos a comenzar con un ejemplo muy sencillo.

Supongamos que queremos construir un sistema que determine si un estudiante **aprobará** o **no aprobará** un examen.

Para cada estudiante conocemos:

- Horas de estudio
- Cantidad de ejercicios resueltos

Y también sabemos el resultado final.

## Mostrar gráfico

```
Ejercicios

5 |                     ●
4 |
3 |             ●
2 |
1 |  ○

   +------------------------
      1      2      3      4
           Horas

○ No aprobó
● Aprobó
```

Explicar:

Cada punto representa un estudiante.

Todavía no existe ningún modelo.

Solamente tenemos ejemplos.

---

# ¿Qué hace Machine Learning?

El algoritmo observa todos estos ejemplos e intenta encontrar un patrón que permita separar ambas categorías.

## Mostrar gráfico

```
Ejercicios

5 |                     ●
4 |
3 |---------------------------
2 |
1 |  ○

   +------------------------
```

Explicar:

El algoritmo busca una frontera que permita distinguir ambas clases.

Una vez encontrada esa frontera podremos clasificar nuevos estudiantes.

---

# Clasificando un nuevo estudiante

Supongamos que aparece un estudiante nuevo.

## Mostrar gráfico

```
Ejercicios

5 |                     ●
4 |
3 |---------------------------
2 |            ★
1 |  ○

   +------------------------
```

Preguntar a la clase:

**¿De qué lado quedó?**

Como quedó debajo de la frontera, el modelo predice:

> No aprobará.

Eso es una clasificación.

---

# Ejemplo 2: Predicción

Machine Learning también puede utilizarse para predecir valores numéricos.

Por ejemplo:

Queremos estimar el precio de una casa.

Conocemos:

- Metros cuadrados
- Precio

## Mostrar gráfico

```
Precio

350 |
300 |                  ●
250 |
200 |           ●
150 |     ●

   +------------------------
     50   80  120  150

        Metros cuadrados
```

El algoritmo ajusta una línea que representa la tendencia.

---

# Realizando una predicción

Supongamos ahora que aparece una casa de 100 m².

## Mostrar gráfico

```
Precio

350 |
300 |                 ●
250 |              /
200 |          ●  /
150 |      ●  /
100 |       ★

   +------------------------
```

La línea permite estimar aproximadamente el precio de esa nueva vivienda.

Esto ya no es clasificación.

Es una predicción numérica (regresión).

---

# Idea principal

En términos generales:

Machine Learning sirve para realizar:

- Clasificaciones
- Predicciones

Prácticamente todos los algoritmos buscan resolver uno de estos dos problemas.

---

# Datos de entrenamiento

Hasta ahora usamos únicamente ejemplos.

Esos ejemplos reciben un nombre.

Se llaman:

> **Training Data**

Son los datos utilizados para entrenar el modelo.

## Mostrar gráfico

```
Training Data

●   ●   ●

○   ○   ○
```

El algoritmo aprende únicamente observando estos datos.

---

# ¿Cómo sabemos si el modelo realmente aprendió?

Acá aparece un problema.

Supongamos que entrenamos dos modelos.

Modelo A:

Una línea simple.

Modelo B:

Una curva extremadamente compleja que pasa exactamente por todos los puntos.

## Mostrar gráfico

Modelo A

```
●

      ●

           ○

---------------
```

Modelo B

```
●~~~~~

     ●~~~~~~

           ○~~~~~
```

A simple vista parecería que el segundo modelo es mejor.

Después de todo, pasa exactamente por todos los puntos.

Pero...

¿Realmente aprendió?

---

# Datos de prueba

Para responder esa pregunta utilizizamos otro conjunto de datos.

Estos datos nunca fueron utilizados durante el entrenamiento.

Se llaman:

> **Testing Data**

## Mostrar gráfico

```
Training

● ● ● ○ ○ ○

↓

Modelo

↓

Testing

● ○ ●
```

---

# Evaluando ambos modelos

Ahora hacemos que ambos modelos intenten predecir los datos de Testing.

Modelo A

```
Predicción

● ✔

○ ✔

● ✔
```

Modelo B

```
Predicción

● ✖

○ ✔

● ✖
```

Explicar:

Aunque el segundo modelo aprendió perfectamente los datos de entrenamiento, falla cuando aparecen datos nuevos.

---

# ¿Qué ocurrió?

El segundo modelo memorizó los ejemplos.

No aprendió el patrón general.

Cuando llegaron datos nuevos comenzó a equivocarse.

Este fenómeno recibe el nombre de:

> **Overfitting**

Lo veremos con mucho más detalle en otra clase.

Por ahora basta con comprender la idea.

---

# Una enseñanza muy importante

Cuando trabajamos con Machine Learning...

No gana el modelo que mejor funciona con Training.

Gana el modelo que mejor funciona con Testing.

Porque nuestro objetivo es predecir correctamente datos que todavía no conocemos.

---

# ¿Importa el algoritmo?

Hoy existen muchísimos algoritmos.

- Árboles de decisión
- KNN
- Regresión logística
- Random Forest
- SVM
- Redes neuronales
- XGBoost

Todos funcionan de manera distinta.

Pero todos persiguen exactamente el mismo objetivo:

Realizar buenas predicciones sobre datos nuevos.

Por eso siempre los evaluamos utilizando Testing Data.

---

# Resumen

Machine Learning consiste en entrenar algoritmos utilizando ejemplos.

Los algoritmos pueden utilizarse para:

- Clasificar
- Predecir

Entrenan utilizando **Training Data**.

Se evalúan utilizando **Testing Data**.

Un modelo que memoriza los ejemplos no necesariamente será un buen modelo.

Lo importante no es qué tan bien aprende el entrenamiento, sino qué tan bien funciona con datos que nunca había visto.
