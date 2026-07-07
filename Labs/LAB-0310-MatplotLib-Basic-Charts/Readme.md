# Laboratorio 0310 – Gráficos característicos de Machine Learning

## Objetivo

Conocer los 4 gráficos más usados en Machine Learning, entender qué muestra cada uno y por qué se usa. Vamos a trabajar con datos simples creados a mano, sin usar librerías externas de datasets.

- Histograma
- Boxplot
- Scatter
- Heatmap de correlación

## Requisitos previos

Tener instalado Python con las librerías `matplotlib` y `numpy`. Si no las tenés instaladas, ejecutar en una terminal:

```
pip install matplotlib numpy
```

## Paso 0 – Importar las librerías

Al principio de cada script (o celda de notebook) hay que importar las librerías que vamos a usar.

```python
import matplotlib.pyplot as plt
import numpy as np
```

- `import matplotlib.pyplot as plt`: importa la parte de matplotlib que sirve para hacer gráficos, y la renombra como `plt` para escribir menos.
- `import numpy as np`: importa numpy, una librería para trabajar con números y listas de forma más eficiente. La usamos solo en el Ejercicio 4.

Esta línea va una sola vez al principio del archivo, no hace falta repetirla en cada ejercicio.

---

## Ejercicio 1 – Histograma

**¿Qué muestra?** Cómo se distribuyen los valores de una variable: si hay valores repetidos, si se agrupan en rangos, si hay valores raros (outliers).

**¿Para qué sirve en ML?** Antes de entrenar un modelo, es importante saber cómo están distribuidos los datos. Si una variable tiene grupos bien separados, puede ser una buena pista de que existen categorías distintas en los datos.

### Código

```python
# Datos: edades de 15 personas, inventadas a mano
edades = [22, 23, 25, 25, 26, 30, 31, 32, 45, 46, 47, 48, 60, 61, 62]

plt.hist(edades, bins=5)
plt.title("Distribución de edades")
plt.xlabel("Edad")
plt.ylabel("Cantidad de personas")
plt.show()
```

### Explicación línea por línea

- `edades = [22, 23, 25, ...]`: creamos una lista de Python con 15 números, inventados a mano, que representan edades.
- `plt.hist(edades, bins=5)`: dibuja el histograma.
  - El primer argumento (`edades`) es la lista de datos que queremos graficar.
  - `bins=5` significa que el eje X se va a dividir en 5 rangos (por ejemplo: 22-30, 30-38, 38-46, etc.). Cada barra del histograma muestra cuántos valores caen dentro de ese rango.
- `plt.title("Distribución de edades")`: pone un título arriba del gráfico.
- `plt.xlabel("Edad")`: pone el nombre del eje horizontal (X).
- `plt.ylabel("Cantidad de personas")`: pone el nombre del eje vertical (Y).
- `plt.show()`: muestra el gráfico en pantalla. Sin esta línea, el gráfico no aparece.

### Consigna

Mirando el histograma:
1. ¿Se ven grupos de edades separados (jóvenes, adultos, mayores)? Sí: se ven 3 grupos claros, uno cerca de los 22-32 años, otro cerca de los 45-48 y otro cerca de los 60-62, con espacios vacíos entre medio.
2. ¿Por qué te parece que esto es útil en Machine Learning? Pensá en un caso donde quisieras clasificar personas por grupo etario. Es útil porque si los grupos ya se ven separados en el histograma, es una señal de que un modelo de clasificación (por ejemplo, "joven", "adulto", "mayor") va a poder distinguirlos con relativa facilidad, ya que los datos no están mezclados entre sí.

---

## Ejercicio 2 – Boxplot

**¿Qué muestra?** Un resumen de una variable: el valor mínimo, el máximo, la mediana (el valor del medio), y si hay outliers (valores muy alejados del resto). Sirve especialmente para comparar dos o más grupos entre sí.

**¿Para qué sirve en ML?** Antes de entrenar un modelo que compare grupos (por ejemplo, dos cursos, dos sucursales, dos categorías de clientes), conviene ver si esos grupos son realmente distintos o parecidos.

### Código

```python
# Notas de examen de dos cursos, inventadas a mano
curso_a = [6, 7, 7, 8, 8, 9, 9, 10]
curso_b = [3, 4, 5, 5, 6, 6, 7, 9]

plt.boxplot([curso_a, curso_b], labels=["Curso A", "Curso B"])
plt.title("Notas por curso")
plt.ylabel("Nota")
plt.show()
```

### Explicación línea por línea

- `curso_a = [6, 7, 7, 8, ...]` y `curso_b = [3, 4, 5, 5, ...]`: dos listas con las notas de cada curso, inventadas a mano.
- `plt.boxplot([curso_a, curso_b], labels=["Curso A", "Curso B"])`: dibuja una "caja" (boxplot) por cada lista que le pasamos.
  - El primer argumento es una lista de listas: `[curso_a, curso_b]`. Cada lista interna se convierte en una caja del gráfico.
  - `labels=["Curso A", "Curso B"]`: pone el nombre debajo de cada caja, en el mismo orden en que pusimos los datos.
- `plt.title("Notas por curso")`: título del gráfico.
- `plt.ylabel("Nota")`: nombre del eje vertical.
- `plt.show()`: muestra el gráfico.

### Cómo leer la caja

Cada caja tiene una línea en el medio (la mediana), los bordes de la caja (donde está el 50% central de los datos) y unas líneas que salen arriba y abajo (los valores más extremos, sin contar outliers). Los puntos sueltos fuera de esas líneas, si aparecen, son outliers.

### Consigna

1. ¿Qué curso tiene notas más parejas (caja más chica)? El Curso A, porque sus notas van de 6 a 10, un rango más chico.
2. ¿Cuál tiene más dispersión (caja más grande)? El Curso B, porque sus notas van de 3 a 9, un rango más amplio.
3. ¿Por qué en ML es útil comparar grupos así, antes de entrenar un modelo? Porque permite ver si dos grupos son realmente distintos o se superponen. Si un grupo tiene mucha dispersión, puede ser más difícil de predecir o puede necesitar más datos para que el modelo aprenda bien su comportamiento.

---

## Ejercicio 3 – Scatter (gráfico de dispersión)

**¿Qué muestra?** La relación entre dos variables. Cada punto representa un caso, ubicado según sus dos valores.

**¿Para qué sirve en ML?** Sirve para ver a simple vista si dos variables están relacionadas (por ejemplo, si una sube cuando la otra sube), lo cual es una pista de que un modelo podría aprender a predecir una a partir de la otra.

### Código

```python
# Horas de estudio y nota obtenida, inventadas a mano
horas_estudio = [1, 2, 2, 3, 4, 5, 6, 7, 8, 9]
nota_final    = [3, 4, 5, 5, 6, 6, 7, 8, 9, 10]

plt.scatter(horas_estudio, nota_final)
plt.title("Horas de estudio vs Nota")
plt.xlabel("Horas de estudio")
plt.ylabel("Nota final")
plt.show()
```

### Explicación línea por línea

- `horas_estudio = [1, 2, 2, ...]`: lista con las horas que estudió cada persona.
- `nota_final = [3, 4, 5, ...]`: lista con la nota que sacó cada persona. Importante: la posición 0 de esta lista corresponde a la misma persona que la posición 0 de `horas_estudio`, y así sucesivamente.
- `plt.scatter(horas_estudio, nota_final)`: dibuja un punto por cada persona.
  - El primer argumento (`horas_estudio`) define la posición en el eje X de cada punto.
  - El segundo argumento (`nota_final`) define la posición en el eje Y.
- `plt.title(...)`, `plt.xlabel(...)`, `plt.ylabel(...)`, `plt.show()`: igual que en los ejercicios anteriores.

### Consigna

1. ¿Se ve una relación clara entre las dos variables (a más horas, más nota)? Sí, los puntos forman una tendencia ascendente: a medida que aumentan las horas de estudio, la nota también aumenta.
2. ¿Te parece que un modelo de Machine Learning podría predecir la nota final a partir de las horas de estudio? Sí, porque la relación es bastante lineal y consistente, lo que facilita que un modelo de regresión aprenda ese patrón.

---

## Ejercicio 4 – Heatmap de correlación

**¿Qué muestra?** Qué tan relacionadas están varias variables entre sí, todas juntas en una sola imagen, usando colores.

**¿Para qué sirve en ML?** Cuando tenemos muchas variables (no solo dos), este gráfico ayuda a elegir cuáles conviene usar en un modelo, y a detectar variables que dicen "lo mismo" (muy correlacionadas entre sí).

### Código

```python
# 3 variables inventadas a mano, mismas personas en cada lista
horas_estudio = [1, 2, 3, 4, 5, 6, 7, 8]
horas_sueño   = [8, 7, 7, 6, 6, 5, 5, 4]
nota_final    = [3, 4, 5, 6, 7, 8, 9, 10]

# Juntamos las 3 listas en una sola estructura
datos = np.array([horas_estudio, horas_sueño, nota_final])

# Calculamos qué tan relacionada está cada variable con las demás
matriz_correlacion = np.corrcoef(datos)

# Creamos el "lienzo" y el "área de dibujo"
figura, eje = plt.subplots()

# Dibujamos la matriz de correlación como una imagen de colores
imagen = eje.imshow(matriz_correlacion, cmap="coolwarm", vmin=-1, vmax=1)

# Ponemos los nombres de las variables en los ejes
nombres = ["Horas estudio", "Horas sueño", "Nota final"]
eje.set_xticks([0, 1, 2])
eje.set_xticklabels(nombres, rotation=45, ha="right")
eje.set_yticks([0, 1, 2])
eje.set_yticklabels(nombres)

# Agregamos la barra de colores al costado
plt.colorbar(imagen)
plt.title("Correlación entre variables")
plt.show()
```

### Explicación línea por línea

- `horas_estudio`, `horas_sueño`, `nota_final`: tres listas de datos inventados. El orden importa: la posición 0 de cada lista corresponde a la misma persona en las tres.
- `datos = np.array([horas_estudio, horas_sueño, nota_final])`: junta las 3 listas en una sola estructura de numpy, para poder hacer cálculos matemáticos sobre ellas.
- `matriz_correlacion = np.corrcoef(datos)`: calcula, para cada par de variables, un número entre -1 y 1 que indica qué tan relacionadas están.
  - Cerca de 1: cuando una sube, la otra también sube.
  - Cerca de -1: cuando una sube, la otra baja.
  - Cerca de 0: no hay relación clara.
- `figura, eje = plt.subplots()`: crea dos cosas a la vez.
  - `figura`: es el "lienzo" completo, la ventana del gráfico.
  - `eje`: es el "área de dibujo" dentro de esa ventana, donde efectivamente vamos a dibujar. En los ejercicios anteriores usamos `plt` directamente porque solo había un gráfico simple; acá usamos `eje` porque necesitamos controlar más detalles (los nombres de los ejes, por ejemplo).
- `imagen = eje.imshow(matriz_correlacion, cmap="coolwarm", vmin=-1, vmax=1)`: dibuja la matriz de correlación como una imagen, donde cada celda tiene un color según su valor.
  - `cmap="coolwarm"`: define la paleta de colores a usar (tonos de azul y rojo).
  - `vmin=-1, vmax=1`: le decimos que los colores deben representar valores entre -1 y 1, que es el rango posible de la correlación.
- `nombres = [...]`: lista con los nombres de las 3 variables, en el mismo orden en que las pusimos en `datos`.
- `eje.set_xticks([0, 1, 2])`: indica que en el eje X va a haber 3 marcas, en las posiciones 0, 1 y 2 (una por variable).
- `eje.set_xticklabels(nombres, rotation=45, ha="right")`: pone el nombre de cada variable en esas marcas del eje X. `rotation=45` las rota 45 grados para que no se superpongan entre sí.
- `eje.set_yticks([0, 1, 2])` y `eje.set_yticklabels(nombres)`: lo mismo pero para el eje Y.
- `plt.colorbar(imagen)`: agrega al costado del gráfico una barra que muestra qué color corresponde a qué valor de correlación.
- `plt.title(...)` y `plt.show()`: como en los ejercicios anteriores.

### Consigna

1. ¿Qué par de variables está más correlacionado (color más intenso, cerca de 1 o -1)? "Horas de estudio" y "Nota final" están muy correlacionadas cerca de 1, y "Horas de estudio" con "Horas de sueño" están muy correlacionadas cerca de -1.
2. ¿Tiene sentido ese resultado con lo que esperarías en la realidad? Sí: en estos datos, a más horas de estudio, mejor nota (correlación positiva), y a más horas de estudio, menos horas de sueño (correlación negativa), lo cual es consistente con la lógica de los datos inventados.

---

## Cierre del laboratorio

Responder por escrito: de estos 4 gráficos (histograma, boxplot, scatter, heatmap), ¿cuál usarías primero al recibir un conjunto de datos nuevo, y por qué? El histograma, porque da una primera idea rápida de cómo está distribuida cada variable de forma individual, antes de empezar a cruzarla con otras variables o compararla entre grupos.
