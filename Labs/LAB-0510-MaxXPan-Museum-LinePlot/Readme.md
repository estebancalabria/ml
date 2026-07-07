# LAB-0510-MatxPan-Museo-Line-Charts

## Objetivo

Combinar pandas y matplotlib: cargar un dataset real desde un CSV, explorarlo con los métodos básicos de pandas, y graficar su evolución en el tiempo con gráficos de líneas.

## Requisitos previos

Tener instalado Python con `pandas`, `matplotlib` y `seaborn`.

```
pip install pandas matplotlib seaborn
```

## El dataset

El archivo tiene la cantidad de visitantes mensuales a 4 museos, entre 2014 y 2018. Así se ven las primeras filas:

```
Date,Avila Adobe,Firehouse Museum,Chinese American Museum,America Tropical Interpretive Center
2014-01-01,24778,4486,1581,6602
2014-02-01,18976,4172,1785,5029
2014-03-01,25231,7082,3229,8129
```

- `Date`: primer día del mes al que corresponde el dato.
- Cada columna siguiente es un museo, con la cantidad de visitantes de ese mes.

El dataset completo está en:
`https://raw.githubusercontent.com/estebancalabria/ml/refs/heads/main/datasets/museum_visitors.csv`

---

## Ejercicio 1 – Cargar el dataset

### Código

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://raw.githubusercontent.com/estebancalabria/ml/refs/heads/main/datasets/museum_visitors.csv"

museos = pd.read_csv(url, index_col="Date", parse_dates=True)

museos.head()
```

### Explicación línea por línea

- `pd.read_csv(url, index_col="Date", parse_dates=True)`: lee el CSV desde la URL y lo carga como DataFrame.
  - `index_col="Date"`: usa la columna `Date` como índice, en vez de dejarla como una columna más.
  - `parse_dates=True`: convierte esa columna en fechas reales, para que el eje del tiempo se ordene bien al graficar.
- `museos.head()`: muestra las primeras 5 filas, para chequear que se cargó bien.

### Consigna

Corré `museos.shape` y `museos.columns`. ¿Cuántas filas (meses) tiene el dataset y cuántos museos hay?

---

## Ejercicio 2 – Explorar el dataset

### Código

```python
museos.info()
museos.describe()
```

### Explicación línea por línea

- `museos.info()`: muestra el tipo de dato de cada columna y si hay valores nulos.
- `museos.describe()`: muestra estadísticas básicas (promedio, mínimo, máximo, etc.) de cada museo.

### Consigna

Según `describe()`, ¿qué museo tiene el promedio de visitantes más alto?

---

## Ejercicio 3 – Gráfico de líneas de todos los museos, con seaborn

### Código

```python
plt.figure(figsize=(16, 6))

sns.lineplot(data=museos)

plt.title("Visitantes mensuales por museo")
plt.xlabel("Fecha")
plt.ylabel("Cantidad de visitantes")
plt.show()
```

### Explicación línea por línea

- `plt.figure(figsize=(16, 6))`: define el tamaño del gráfico.
- `sns.lineplot(data=museos)`: dibuja una línea por cada columna del DataFrame, usando el índice (las fechas) como eje X. Seaborn asigna un color distinto a cada museo y agrega la leyenda automáticamente.
- El resto (`title`, `xlabel`, `ylabel`, `show`) es igual a lo visto en laboratorios anteriores.

### Consigna

Mirando el gráfico: ¿algún museo muestra un patrón que se repite todos los años (estacionalidad), con subas y bajas en los mismos meses?

---

## Ejercicio 4 – El mismo gráfico sin seaborn

### Código

```python
plt.figure(figsize=(16, 6))

for museo in museos.columns:
    plt.plot(museos.index, museos[museo], label=museo)

plt.title("Visitantes mensuales por museo")
plt.xlabel("Fecha")
plt.ylabel("Cantidad de visitantes")
plt.legend()
plt.show()
```

### Explicación línea por línea

- `for museo in museos.columns:`: recorre cada columna (cada museo) del DataFrame.
- `plt.plot(museos.index, museos[museo], label=museo)`: dibuja una línea por museo, usando las fechas como eje X y los visitantes de esa columna como eje Y.
- `plt.legend()`: hay que agregarla a mano, a diferencia de `sns.lineplot` que la incluye sola.

### Consigna

Compará este código con el del Ejercicio 3. ¿Cuántas líneas de código extra hacen falta sin seaborn?

---

## Ejercicio 5 – Gráfico de un solo museo

### Código

```python
plt.figure(figsize=(16, 6))

plt.plot(museos.index, museos["Avila Adobe"])

plt.title("Visitantes mensuales - Avila Adobe")
plt.xlabel("Fecha")
plt.ylabel("Cantidad de visitantes")
plt.show()
```

### Explicación línea por línea

- `museos["Avila Adobe"]`: selecciona una sola columna del DataFrame (una Series), la del museo Avila Adobe.
- `plt.plot(museos.index, museos["Avila Adobe"])`: al graficar una sola serie no hace falta `label` ni `plt.legend()`, porque no hay que distinguir entre varias líneas.

### Consigna

Repetí el gráfico para el museo `"Firehouse Museum"`, cambiando el nombre de la columna y el título.

---

## Cierre del laboratorio

Con `pd.read_csv` cargaste un dataset real, con `info()` y `describe()` lo exploraste, y con `sns.lineplot` (o `plt.plot` en un `for`) graficaste la evolución de varias series en el tiempo. Esta combinación de pandas + matplotlib/seaborn es la base de casi cualquier análisis exploratorio de series temporales.
