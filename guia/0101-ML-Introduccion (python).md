# Clase: Introducción Suave al Machine Learning con Python

## Objetivos

Al finalizar esta clase los estudiantes serán capaces de:

* Comprender qué es Machine Learning.
* Diferenciar clasificación y regresión.
* Entender el propósito de los datos de entrenamiento y prueba.
* Comprender por qué un modelo muy complejo no necesariamente es mejor.
* Entender la idea de generalización.
* Comprender la importancia de evaluar modelos con datos nuevos.
* Visualizar con Python la diferencia entre una recta simple y una curva compleja.
* Calcular errores de predicción de manera intuitiva.

***

## Secuencia

```text
Árbol de decisión
        ↓
Clasificación
        ↓
Regresión (línea recta)
        ↓
Training Data
        ↓
Línea recta vs curva compleja
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

# Preparación del entorno

Antes de empezar, importamos las librerías que vamos a usar.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# Código 1 — Simulando un árbol de decisión simple

En este primer bloque no vamos a entrenar todavía un modelo real.

Vamos a escribir una función que imite el comportamiento de un árbol de decisión.

```python
def predecir_si_le_gusta_el_curso(le_gusta_programar, le_gusta_resolver_problemas):
    if le_gusta_programar == "Sí":
        if le_gusta_resolver_problemas == "Sí":
            return "Le gustará el curso"
        else:
            return "No le gustará el curso"
    else:
        return "No le gustará el curso"
```

Probamos algunos casos.

```python
predecir_si_le_gusta_el_curso("Sí", "Sí")
```

```python
predecir_si_le_gusta_el_curso("Sí", "No")
```

```python
predecir_si_le_gusta_el_curso("No", "Sí")
```

***

## Explicación para la clase

Este código representa la misma idea que el árbol.

No devuelve un número.

Devuelve una categoría:

* Le gustará el curso
* No le gustará el curso

Eso es una clasificación.

***

## Primera idea importante

Este árbol es una forma simple de razonar como lo haría un modelo.

Más adelante, en Machine Learning real, el árbol no lo escribimos nosotros a mano.

El algoritmo aprende esas reglas a partir de datos.

***

# Parte 2 — Predicción Numérica: Regresión

Veamos ahora otro problema.

Supongamos que disponemos de información de alumnos anteriores.

Para cada alumno conocemos:

* Horas de estudio
* Nota obtenida

***

# Código 2 — Crear datos de alumnos

Creamos un pequeño conjunto de datos inventado.

```python
datos = pd.DataFrame({
    "horas_estudio": [1, 2, 3, 4, 5, 6],
    "nota": [2.5, 4.2, 5.0, 6.8, 7.4, 9.0]
})

datos
```

***

## Mostrar gráfico

```python
plt.scatter(datos["horas_estudio"], datos["nota"])

plt.xlabel("Horas de estudio")
plt.ylabel("Nota")
plt.title("Horas de estudio vs nota obtenida")

plt.ylim(0, 10)
plt.grid(True)
plt.show()
```

***

## ¿Qué observamos?

A medida que aumentan las horas de estudio:

→ La nota tiende a aumentar.

Existe una tendencia.

Cada punto representa un alumno.

Todavía no hay un modelo.

Solamente tenemos datos.

***

# Parte 3 — Ajustando una línea

Podemos representar esa tendencia mediante una línea.

Para eso vamos a ajustar una recta a los datos.

La recta intentará resumir la relación entre:

* horas de estudio
* nota obtenida

***

# Código 3 — Ajustar una recta

Usamos `numpy.polyfit` con grado `1`.

Grado `1` significa:

> una recta.

```python
x = datos["horas_estudio"]
y = datos["nota"]

coeficientes_recta = np.polyfit(x, y, deg=1)

coeficientes_recta
```

Creamos una función para usar esa recta.

```python
modelo_recta = np.poly1d(coeficientes_recta)

modelo_recta
```

***

## Graficar la recta

```python
x_linea = np.linspace(1, 6, 100)
y_linea = modelo_recta(x_linea)

plt.scatter(datos["horas_estudio"], datos["nota"], label="Training Data")
plt.plot(x_linea, y_linea, color="black", label="Modelo: línea recta")

plt.xlabel("Horas de estudio")
plt.ylabel("Nota")
plt.title("Modelo simple: línea recta")

plt.ylim(0, 10)
plt.grid(True)
plt.legend()
plt.show()
```

***

## Explicación para la clase

La línea intenta capturar la tendencia general.

No pasa exactamente por todos los puntos.

Pero resume bastante bien la relación general:

> A más horas de estudio, mayor nota esperada.

***

# Parte 4 — Realizando una predicción

Aparece un alumno nuevo.

Sabemos que estudió 4 horas.

Queremos estimar qué nota podría obtener.

***

# Código 4 — Predecir la nota de un nuevo alumno

```python
horas_nuevo_alumno = 4

nota_predicha = modelo_recta(horas_nuevo_alumno)

nota_predicha
```

Podemos mostrarlo sobre el gráfico.

```python
plt.scatter(datos["horas_estudio"], datos["nota"], label="Training Data")
plt.plot(x_linea, y_linea, color="black", label="Modelo: línea recta")

plt.scatter(horas_nuevo_alumno, nota_predicha, color="red", s=120, marker="*", label="Nuevo alumno")

plt.xlabel("Horas de estudio")
plt.ylabel("Nota")
plt.title("Predicción para un nuevo alumno")

plt.ylim(0, 10)
plt.grid(True)
plt.legend()
plt.show()
```

***

## Explicación

El modelo predice una nota aproximada para un alumno que estudió 4 horas.

La salida es un número.

Por ejemplo:

```text
Nota aproximada = 6.4
```

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
* Predecir valores numéricos

***

# Parte 5 — Datos de Entrenamiento

Los ejemplos que usamos para construir un modelo reciben un nombre.

# Training Data

o

# Datos de Entrenamiento

En nuestro ejemplo:

Los alumnos utilizados para dibujar la línea son Training Data.

***

# Código 5 — Marcar estos datos como Training Data

```python
training_data = datos.copy()

training_data
```

***

## ¿Qué hace el modelo?

Observa esos ejemplos.

Y aprende patrones.

En este caso, aprendió una relación aproximada entre:

```text
horas de estudio → nota
```

***

# Parte 6 — Dos Modelos Distintos

Supongamos ahora que entrenamos dos modelos diferentes.

## Modelo A

Una línea simple.

## Modelo B

Una curva muy compleja.

El objetivo ahora es reproducir con Python la idea del vídeo:

> La curva compleja puede ajustarse mejor al Training Data, pero eso no significa que prediga mejor datos nuevos.

***

# Código 6 — Modelo A: línea recta

Ya tenemos el modelo de línea recta.

```python
modelo_recta
```

***

# Código 7 — Modelo B: curva compleja

Ahora ajustamos una curva de grado alto.

Como tenemos 6 puntos, vamos a usar una curva de grado 5.

Esto permite que la curva pase prácticamente por todos los puntos de entrenamiento.

```python
coeficientes_curva = np.polyfit(x, y, deg=5)

modelo_curva = np.poly1d(coeficientes_curva)

modelo_curva
```

***

## Graficar línea recta vs curva compleja

```python
x_grafico = np.linspace(1, 6, 300)

y_recta = modelo_recta(x_grafico)
y_curva = modelo_curva(x_grafico)

plt.scatter(training_data["horas_estudio"], training_data["nota"], label="Training Data")

plt.plot(x_grafico, y_recta, color="black", label="Modelo A: línea recta")
plt.plot(x_grafico, y_curva, color="green", label="Modelo B: curva compleja")

plt.xlabel("Horas de estudio")
plt.ylabel("Nota")
plt.title("Línea recta vs curva compleja")

plt.ylim(0, 10)
plt.grid(True)
plt.legend()
plt.show()
```

***

## Pregunta para la clase

¿Cuál parece mejor?

La mayoría probablemente responderá:

> La curva verde.

Porque pasa mucho más cerca de los puntos.

Incluso puede parecer que aprendió perfectamente.

***

# Pero...

Recordemos algo.

El objetivo del Machine Learning no es explicar perfectamente los datos antiguos.

El objetivo es:

> Realizar predicciones sobre datos nuevos.

***

# Parte 7 — Datos de Prueba

Necesitamos nuevos alumnos.

Alumnos que no participaron del entrenamiento.

Estos datos reciben el nombre de:

# Testing Data

o

# Datos de Prueba

***

# Código 8 — Crear Testing Data

Estos datos representan alumnos nuevos.

El modelo no los vio cuando se ajustó la recta ni cuando se ajustó la curva.

```python
testing_data = pd.DataFrame({
    "horas_estudio": [1.5, 2.5, 3.5, 4.5, 5.5],
    "nota_real": [3.4, 4.9, 6.1, 7.2, 8.5]
})

testing_data
```

***

## Visualizar Training Data y Testing Data

```python
plt.scatter(training_data["horas_estudio"], training_data["nota"], label="Training Data")
plt.scatter(testing_data["horas_estudio"], testing_data["nota_real"], color="blue", marker="*", s=120, label="Testing Data")

plt.xlabel("Horas de estudio")
plt.ylabel("Nota")
plt.title("Training Data y Testing Data")

plt.ylim(0, 10)
plt.grid(True)
plt.legend()
plt.show()
```

***

## Resumen

Training Data:

```text
Sirve para aprender.
```

Testing Data:

```text
Sirve para evaluar.
```

***

# Parte 8 — Probando la línea recta

Ahora hacemos que el modelo de línea recta prediga la nota de cada alumno del Testing Data.

***

# Código 9 — Predicciones de la recta

```python
testing_data["prediccion_recta"] = modelo_recta(testing_data["horas_estudio"])

testing_data
```

***

## Calcular errores de la recta

El error será la distancia entre:

```text
nota real - nota predicha
```

Para simplificar, usamos el valor absoluto.

```python
testing_data["error_recta"] = abs(testing_data["nota_real"] - testing_data["prediccion_recta"])

testing_data
```

***

## Error total de la recta

```python
error_total_recta = testing_data["error_recta"].sum()

error_total_recta
```

***

## Visualizar errores de la recta

```python
plt.scatter(training_data["horas_estudio"], training_data["nota"], label="Training Data")
plt.scatter(testing_data["horas_estudio"], testing_data["nota_real"], color="blue", marker="*", s=120, label="Testing Data")

plt.plot(x_grafico, modelo_recta(x_grafico), color="black", label="Modelo A: línea recta")

for _, fila in testing_data.iterrows():
    plt.plot(
        [fila["horas_estudio"], fila["horas_estudio"]],
        [fila["nota_real"], fila["prediccion_recta"]],
        color="red",
        linestyle="--"
    )

plt.xlabel("Horas de estudio")
plt.ylabel("Nota")
plt.title("Errores de predicción usando la línea recta")

plt.ylim(0, 10)
plt.grid(True)
plt.legend()
plt.show()
```

***

## Explicación

Cada línea roja representa la distancia entre:

* la nota real
* la nota predicha por el modelo

Cuanto más larga la línea roja, mayor el error.

***

# Parte 9 — Probando la curva compleja

Ahora hacemos exactamente lo mismo con la curva verde.

***

# Código 10 — Predicciones de la curva

```python
testing_data["prediccion_curva"] = modelo_curva(testing_data["horas_estudio"])

testing_data
```

***

## Calcular errores de la curva

```python
testing_data["error_curva"] = abs(testing_data["nota_real"] - testing_data["prediccion_curva"])

testing_data
```

***

## Error total de la curva

```python
error_total_curva = testing_data["error_curva"].sum()

error_total_curva
```

***

## Visualizar errores de la curva

```python
plt.scatter(training_data["horas_estudio"], training_data["nota"], label="Training Data")
plt.scatter(testing_data["horas_estudio"], testing_data["nota_real"], color="blue", marker="*", s=120, label="Testing Data")

plt.plot(x_grafico, modelo_curva(x_grafico), color="green", label="Modelo B: curva compleja")

for _, fila in testing_data.iterrows():
    plt.plot(
        [fila["horas_estudio"], fila["horas_estudio"]],
        [fila["nota_real"], fila["prediccion_curva"]],
        color="red",
        linestyle="--"
    )

plt.xlabel("Horas de estudio")
plt.ylabel("Nota")
plt.title("Errores de predicción usando la curva compleja")

plt.ylim(0, 10)
plt.grid(True)
plt.legend()
plt.show()
```

***

# Parte 10 — Comparación final de errores

Ahora comparamos ambos modelos.

***

# Código 11 — Comparar error total

```python
comparacion = pd.DataFrame({
    "modelo": ["Línea recta", "Curva compleja"],
    "error_total": [error_total_recta, error_total_curva]
})

comparacion
```

***

## Gráfico comparativo

```python
plt.bar(comparacion["modelo"], comparacion["error_total"])

plt.ylabel("Error total")
plt.title("Comparación de errores en Testing Data")

plt.grid(axis="y")
plt.show()
```

***

## Resultado esperado

La línea recta debería tener menor error total que la curva compleja.

Conceptualmente:

```text
Modelo A: línea recta     → error menor
Modelo B: curva compleja  → error mayor
```

***

# ¿Qué ocurrió?

La curva compleja aprendió muy bien los datos de entrenamiento.

Pero al aparecer datos nuevos, se equivocó más.

La línea recta no era perfecta en Training Data.

Pero aprendió mejor la tendencia general.

Por eso predijo mejor en Testing Data.

***

# Enseñanza Fundamental

No gana el modelo que mejor funciona con Training Data.

Gana el modelo que mejor funciona con Testing Data.

Esta es una de las ideas más importantes de todo Machine Learning.

***

# Overfitting

Cuando un modelo aprende demasiado los datos de entrenamiento:

* Memoriza ejemplos.
* Aprende ruido.
* Se adapta demasiado a casos particulares.
* Pierde capacidad de generalización.

Decimos que existe:

# Overfitting

***

## Idea intuitiva

```text
Memorizar ≠ Aprender
```

Un modelo útil debe generalizar.

***

# Código 12 — Ver todos los resultados juntos

```python
testing_data
```

Podemos redondear la tabla para verla mejor.

```python
testing_data_redondeado = testing_data.round(2)

testing_data_redondeado
```

***

## Explicación de la tabla

La tabla muestra:

* Horas de estudio.
* Nota real.
* Predicción de la recta.
* Error de la recta.
* Predicción de la curva.
* Error de la curva.

Así podemos ver que el modelo más complejo no necesariamente es el más útil.

***

# Parte 11 — Visualización completa

Ahora mostramos todo junto:

* Training Data.
* Testing Data.
* Línea recta.
* Curva compleja.

***

# Código 13 — Gráfico completo

```python
plt.scatter(training_data["horas_estudio"], training_data["nota"], label="Training Data")
plt.scatter(testing_data["horas_estudio"], testing_data["nota_real"], color="blue", marker="*", s=120, label="Testing Data")

plt.plot(x_grafico, modelo_recta(x_grafico), color="black", label="Modelo A: línea recta")
plt.plot(x_grafico, modelo_curva(x_grafico), color="green", label="Modelo B: curva compleja")

plt.xlabel("Horas de estudio")
plt.ylabel("Nota")
plt.title("Comparación final: línea recta vs curva compleja")

plt.ylim(0, 10)
plt.grid(True)
plt.legend()
plt.show()
```

***

## Pregunta para la clase

A simple vista, la curva verde parecía mejor porque se acomodaba muy bien al Training Data.

Pero después de evaluar con Testing Data:

> ¿Cuál modelo conviene elegir?

Respuesta esperada:

> La línea recta, porque predice mejor datos nuevos.

***

# Parte 12 — Volviendo al Árbol de Decisión

Recordemos el árbol del principio.

Lo construimos usando Training Data.

Ahora vamos a probarlo utilizando Testing Data.

***

# Código 14 — Crear datos para el árbol

Creamos algunos alumnos ficticios.

```python
alumnos_arbol = pd.DataFrame({
    "le_gusta_programar": ["Sí", "Sí", "No", "Sí", "No"],
    "le_gusta_resolver_problemas": ["Sí", "No", "Sí", "Sí", "No"],
    "realidad": [
        "Le gustará el curso",
        "No le gustará el curso",
        "No le gustará el curso",
        "Le gustará el curso",
        "Le gustará el curso"
    ]
})

alumnos_arbol
```

***

## Usar el árbol para predecir

```python
alumnos_arbol["prediccion"] = alumnos_arbol.apply(
    lambda fila: predecir_si_le_gusta_el_curso(
        fila["le_gusta_programar"],
        fila["le_gusta_resolver_problemas"]
    ),
    axis=1
)

alumnos_arbol
```

***

## Evaluar si acertó o no

```python
alumnos_arbol["acierto"] = alumnos_arbol["prediccion"] == alumnos_arbol["realidad"]

alumnos_arbol
```

***

## Calcular cantidad de aciertos

```python
cantidad_aciertos = alumnos_arbol["acierto"].sum()
cantidad_total = len(alumnos_arbol)

cantidad_aciertos, cantidad_total
```

***

## Calcular accuracy simple

```python
accuracy = cantidad_aciertos / cantidad_total

accuracy
```

***

## Explicación

Acá hicimos con el árbol lo mismo que hicimos con la línea y la curva.

Comparamos:

```text
Predicción vs realidad
```

Si coincide, el modelo acertó.

Si no coincide, el modelo falló.

***

# Parte 13 — ¿Importa qué algoritmo usemos?

Actualmente existen muchísimos algoritmos.

* KNN
* Árboles de decisión
* Regresión logística
* Random Forest
* SVM
* Gradient Boosting
* Redes neuronales

Todos son distintos.

Pero todos se evalúan con la misma idea fundamental:

> Comparar predicciones contra datos reales que el modelo no vio durante el entrenamiento.

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

# Código final — Resumen de la idea principal

```python
print("Machine Learning no trata de memorizar los datos de entrenamiento.")
print("Machine Learning trata de generalizar para predecir datos nuevos.")
print()
print(f"Error total de la línea recta: {error_total_recta:.2f}")
print(f"Error total de la curva compleja: {error_total_curva:.2f}")

if error_total_recta < error_total_curva:
    print("En este ejemplo, la línea recta generaliza mejor.")
else:
    print("En este ejemplo, la curva compleja generaliza mejor.")
```

***

## Mensaje de cierre

> Machine Learning no trata de encontrar el modelo más complejo.
>
> Trata de encontrar el modelo que mejor predice datos que todavía no conocemos.

***

# Nota didáctica para el instructor

Esta clase no busca enseñar todavía modelos formales de Machine Learning.

Busca instalar estas ideas:

1. Un modelo puede clasificar.
2. Un modelo puede predecir valores numéricos.
3. Los datos usados para aprender son Training Data.
4. Los datos usados para evaluar son Testing Data.
5. Un modelo complejo puede memorizar.
6. Memorizar no es lo mismo que aprender.
7. El objetivo real es generalizar.

***

# Versión corta de la explicación oral

Podés decirlo así:

> Primero vimos un árbol que clasificaba personas según si les gustaría o no un curso.
>
> Después vimos una línea que predecía notas según horas de estudio.
>
> Luego comparamos una línea simple contra una curva muy compleja.
>
> La curva parecía mejor porque pasaba por los puntos de entrenamiento.
>
> Pero cuando usamos datos nuevos, la línea cometió menos error.
>
> Entonces aprendimos una idea fundamental: en Machine Learning no gana el modelo que mejor memoriza, sino el que mejor predice datos nuevos.

***

# Variante opcional para cerrar con pregunta

Podés cerrar preguntando:

```text
Si un modelo tiene 100% de acierto en Training Data,
pero falla mucho en Testing Data...

¿Es un buen modelo?
```

Respuesta esperada:

```text
No.

Probablemente memorizó.
Probablemente hizo overfitting.
No generaliza bien.
```

***

Esta versión ya te queda mucho más cercana a una clase práctica tipo “segundo monitor”: seguís la narrativa del vídeo, pero cada idea importante tiene un bloque de código para que los alumnos vean el concepto funcionando.
