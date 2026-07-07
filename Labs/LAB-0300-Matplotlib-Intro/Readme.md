# Laboratorio 0300 – Introducción a matplotlib

## Objetivo

Conocer la librería matplotlib desde cero: qué es, cómo se instala, y cómo se hace el gráfico más simple posible. Este laboratorio es el paso previo al Laboratorio 2 (gráficos característicos de ML).

## Requisitos previos

Tener instalado Python con la librería `matplotlib`. Si no la tenés instalada, ejecutar en una terminal:

```
pip install matplotlib
```

## ¿Qué es matplotlib?

Es una librería de Python para hacer gráficos (líneas, barras, puntos, histogramas, etc.). Es la más usada para visualizar datos en Python, y muchas otras librerías (como pandas o seaborn) la usan por dentro.

---

## Ejercicio 1 – Tu primer gráfico de líneas

### Código

```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [10, 20, 15, 30, 25]

plt.plot(x, y)
plt.show()
```

### Explicación línea por línea

- `import matplotlib.pyplot as plt`: importa la parte de matplotlib que se usa para graficar, y la renombra como `plt` para escribir menos.
- `x = [1, 2, 3, 4, 5]`: lista de valores para el eje horizontal.
- `y = [10, 20, 15, 30, 25]`: lista de valores para el eje vertical. La posición 0 de `y` corresponde a la posición 0 de `x`, y así con cada una.
- `plt.plot(x, y)`: dibuja una línea que une los puntos (1,10), (2,20), (3,15), (4,30), (5,25).
- `plt.show()`: muestra el gráfico en pantalla. Sin esta línea, el gráfico no aparece.

### Consigna

Cambiá los valores de la lista `y` y volvé a correr el código. ¿Cómo cambia la forma de la línea?

---

## Ejercicio 2 – Agregar título y nombres a los ejes

### Código

```python
import matplotlib.pyplot as plt

dias = [1, 2, 3, 4, 5]
temperatura = [18, 20, 19, 23, 25]

plt.plot(dias, temperatura)
plt.title("Temperatura durante la semana")
plt.xlabel("Día")
plt.ylabel("Temperatura (°C)")
plt.show()
```

### Explicación línea por línea

- `dias` y `temperatura`: dos listas de datos, inventadas a mano. Cada posición de una corresponde a la misma posición en la otra.
- `plt.plot(dias, temperatura)`: dibuja la línea. El primer argumento va al eje X, el segundo al eje Y.
- `plt.title("Temperatura durante la semana")`: pone un título arriba del gráfico.
- `plt.xlabel("Día")`: pone el nombre del eje horizontal.
- `plt.ylabel("Temperatura (°C)")`: pone el nombre del eje vertical.
- `plt.show()`: muestra el gráfico.

### Consigna

Agregá un sexto día con una temperatura a elección, actualizando las dos listas (`dias` y `temperatura`), y volvé a correr el código.

---

## Ejercicio 3 – Dos líneas en el mismo gráfico

### Código

```python
import matplotlib.pyplot as plt

dias = [1, 2, 3, 4, 5]
temperatura_lunes_a_viernes = [18, 20, 19, 23, 25]
temperatura_semana_pasada   = [15, 16, 18, 17, 20]

plt.plot(dias, temperatura_lunes_a_viernes, label="Esta semana")
plt.plot(dias, temperatura_semana_pasada, label="Semana pasada")
plt.title("Comparación de temperaturas")
plt.xlabel("Día")
plt.ylabel("Temperatura (°C)")
plt.legend()
plt.show()
```

### Explicación línea por línea

- `plt.plot(dias, temperatura_lunes_a_viernes, label="Esta semana")`: dibuja la primera línea. El argumento `label` le pone un nombre, que se va a usar en la referencia (leyenda) del gráfico.
- `plt.plot(dias, temperatura_semana_pasada, label="Semana pasada")`: dibuja una segunda línea en el mismo gráfico. Matplotlib acumula todo lo que dibujás hasta que llamás a `plt.show()`.
- `plt.legend()`: muestra la referencia (leyenda) con los nombres que pusimos en `label`, para saber qué línea es cuál.
- El resto (`plt.title`, `plt.xlabel`, `plt.ylabel`, `plt.show`) ya lo vimos en el ejercicio anterior.

### Consigna

Agregá una tercera línea con datos inventados por vos (por ejemplo, "semana que viene, estimada") y su correspondiente `label`.

---

## Cierre del laboratorio

Con lo visto acá (`plt.plot`, `plt.title`, `plt.xlabel`, `plt.ylabel`, `plt.legend`, `plt.show`) ya podés armar un gráfico de líneas básico con referencias. En el Laboratorio 2 vamos a ver otros tipos de gráfico (histograma, boxplot, scatter, heatmap) que se usan específicamente en Machine Learning.
