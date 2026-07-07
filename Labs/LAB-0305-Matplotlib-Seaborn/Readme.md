# Laboratorio 0301 – Introducción a seaborn

## Objetivo

Conocer seaborn: qué es, qué le agrega a matplotlib, y ver un ejemplo comparando el mismo gráfico hecho con las dos librerías. Este laboratorio sigue al Laboratorio 0300 (introducción a matplotlib).

## Requisitos previos

Tener instalado Python con las librerías `matplotlib`, `seaborn` y `pandas` (se usa solo para crear la tabla de datos, no para leer archivos). Si no las tenés instaladas, ejecutar en una terminal:

```
pip install matplotlib seaborn pandas
```

## ¿Qué es seaborn?

Página oficial: https://seaborn.pydata.org/

Seaborn es una librería de Python para hacer gráficos que se construye sobre matplotlib. No lo reemplaza: por dentro sigue usando matplotlib, pero agrega funciones que hacen más simple graficar datos de tablas (como las de pandas).

Lo que seaborn le agrega a matplotlib:

- Con matplotlib, si querés graficar varias columnas de una tabla, tenés que llamar a `plt.plot()` una vez por cada columna. Con seaborn, muchas veces alcanza con una sola línea de código y él arma automáticamente una línea por columna, con colores distintos y leyenda incluida.
- Los gráficos vienen con un estilo visual más prolijo por defecto (grilla de fondo, colores, tipografía), sin tener que configurarlo a mano.
- Tiene funciones pensadas específicamente para trabajar con tablas de pandas, pasándole directamente la tabla como dato.

## El dataset

Para este laboratorio no vamos a leer un archivo, sino crear los datos directamente en código: la nota de 3 alumnos a lo largo de 5 exámenes.

---

## Ejercicio 1 – El mismo gráfico con matplotlib puro

### Código

```python
import matplotlib.pyplot as plt

examenes = [1, 2, 3, 4, 5]
alumno_juan  = [6, 7, 6, 8, 9]
alumno_maria = [8, 8, 9, 9, 10]
alumno_pedro = [5, 6, 5, 7, 6]

plt.figure(figsize=(10, 6))

plt.plot(examenes, alumno_juan, label="Juan")
plt.plot(examenes, alumno_maria, label="María")
plt.plot(examenes, alumno_pedro, label="Pedro")

plt.title("Notas por examen (con matplotlib)")
plt.xlabel("Examen")
plt.ylabel("Nota")
plt.legend()
plt.show()
```

### Explicación línea por línea

- `examenes = [1, 2, 3, 4, 5]`: lista con el número de cada examen, para el eje X.
- `alumno_juan`, `alumno_maria`, `alumno_pedro`: una lista de notas por cada alumno, inventadas a mano. Cada posición corresponde al mismo examen que la lista `examenes`.
- `plt.plot(examenes, alumno_juan, label="Juan")`: dibuja una línea con las notas de Juan. `label="Juan"` le pone nombre a esa línea, para usar después en la leyenda.
- Repetimos `plt.plot(...)` una vez por cada alumno, porque con matplotlib puro no hay una forma de graficar las 3 listas de una sola vez si están en variables separadas.
- `plt.legend()`: muestra la leyenda con los nombres que pusimos en cada `label`.
- El resto ya lo vimos en laboratorios anteriores.

### Consigna

Contá cuántas veces se repite la línea `plt.plot(...)` para lograr el gráfico con los 3 alumnos.

---

## Ejercicio 2 – El mismo gráfico con seaborn

### Código

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

examenes = [1, 2, 3, 4, 5]
notas = pd.DataFrame({
    "Juan": [6, 7, 6, 8, 9],
    "María": [8, 8, 9, 9, 10],
    "Pedro": [5, 6, 5, 7, 6]
}, index=examenes)

plt.figure(figsize=(10, 6))

sns.lineplot(data=notas)

plt.title("Notas por examen (con seaborn)")
plt.xlabel("Examen")
plt.ylabel("Nota")
plt.show()
```

### Explicación línea por línea

- `import seaborn as sns`: importa seaborn y lo renombra como `sns` (convención habitual).
- `notas = pd.DataFrame({...}, index=examenes)`: crea una tabla de pandas con una columna por alumno, usando el diccionario que le pasamos. `index=examenes` hace que el número de examen sea el identificador de cada fila, en vez de un número de fila genérico.
- `sns.lineplot(data=notas)`: le pasamos la tabla completa (`notas`) de una sola vez. Seaborn se encarga de:
  - Usar el índice de la tabla (los exámenes) como eje X.
  - Dibujar una línea por cada columna (cada alumno).
  - Asignar un color distinto a cada línea.
  - Agregar la leyenda automáticamente, sin necesidad de `plt.legend()`.
- Notá que seguimos usando `plt.figure()`, `plt.title()`, `plt.xlabel()`, `plt.ylabel()` y `plt.show()`: seaborn no reemplaza esas funciones de matplotlib, se combina con ellas.

### Consigna

Compará las líneas de código del Ejercicio 1 y el Ejercicio 2. ¿Cuántas líneas menos necesitaste con seaborn para lograr el mismo resultado?

---

## Cierre del laboratorio

Seaborn no es una librería que compite con matplotlib, sino que lo simplifica para casos de uso frecuentes con tablas de datos. En los próximos laboratorios vamos a seguir usando matplotlib para gráficos más manuales (como el heatmap) y seaborn cuando el dato ya está en una tabla de pandas y el gráfico es directo.
