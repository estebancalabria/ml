# Laboratorio – 0310 - Gráfico de líneas con datos reales (Rankings FIFA)

## Objetivo

Aprender a cargar un dataset real desde un archivo CSV y graficarlo como gráfico de líneas, usando pandas, matplotlib y seaborn juntos.

## Requisitos previos

Tener instalado Python con las librerías `pandas`, `matplotlib` y `seaborn`. Si no las tenés instaladas, ejecutar en una terminal:

```
pip install pandas matplotlib seaborn
```

## El dataset

El archivo se llama `fifa.csv` y tiene el ranking mensual de 6 selecciones de fútbol (Argentina, Brasil, España, Francia, Alemania, Italia) a lo largo del tiempo. Así se ven las primeras filas:

```
Date,ARG,BRA,ESP,FRA,GER,ITA
1993-08-08,5.0,8.0,13.0,12.0,1.0,2.0
1993-09-23,12.0,1.0,14.0,7.0,5.0,2.0
1993-10-22,9.0,1.0,7.0,14.0,4.0,3.0
```

- La primera columna (`Date`) es la fecha en que se publicó ese ranking.
- Cada columna siguiente (`ARG`, `BRA`, `ESP`, `FRA`, `GER`, `ITA`) es el puesto en el ranking mundial de esa selección en esa fecha.
- Importante: acá **el número más bajo es mejor**. Por ejemplo, el 08/08/1993 Alemania (`GER`) estaba en el puesto 1, o sea que era el mejor ranking de esa fecha. Argentina (`ARG`) estaba en el puesto 5.

El dataset completo está en:
`https://raw.githubusercontent.com/estebancalabria/ml/refs/heads/main/datasets/fifa.csv`

---

## Ejercicio 1 – Cargar el dataset

### Código

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

fifa_filepath = "https://raw.githubusercontent.com/estebancalabria/ml/refs/heads/main/datasets/fifa.csv"

fifa_data = pd.read_csv(fifa_filepath, index_col="Date", parse_dates=True)

fifa_data.head()
```

### Explicación línea por línea

- `import pandas as pd`: importa pandas, la librería para trabajar con datos en tablas (como un Excel), y la renombra como `pd`.
- `import matplotlib.pyplot as plt`: importa matplotlib para graficar.
- `import seaborn as sns`: importa seaborn, una librería que se construye sobre matplotlib y facilita hacer gráficos más prolijos con menos código.
- `fifa_filepath = "https://..."`: guardamos en una variable la dirección del archivo CSV, para no repetirla.
- `fifa_data = pd.read_csv(fifa_filepath, index_col="Date", parse_dates=True)`: lee el archivo CSV y lo carga como una tabla de pandas.
  - `index_col="Date"`: usa la columna `Date` como identificador de cada fila, en vez de dejarla como una columna más.
  - `parse_dates=True`: le dice a pandas que interprete esa columna como fechas reales (no como texto), para que después el gráfico ordene bien el eje del tiempo.
- `fifa_data.head()`: muestra las primeras 5 filas de la tabla cargada, para chequear que se leyó bien.

### Consigna

Corré el código y mirá la tabla que aparece. ¿Cuántas columnas tiene además de la fecha?

---

## Ejercicio 2 – Graficar el ranking a lo largo del tiempo

### Código

```python
plt.figure(figsize=(16, 6))

sns.lineplot(data=fifa_data)

plt.title("Evolución del ranking FIFA")
plt.xlabel("Fecha")
plt.ylabel("Puesto en el ranking (más bajo es mejor)")
plt.show()
```

### Explicación línea por línea

- `plt.figure(figsize=(16, 6))`: crea el "lienzo" del gráfico, indicando su ancho y alto en pulgadas (16 de ancho, 6 de alto). Esto se hace antes de graficar, para que el gráfico tenga ese tamaño.
- `sns.lineplot(data=fifa_data)`: dibuja una línea por cada columna de la tabla (`ARG`, `BRA`, `ESP`, `FRA`, `GER`, `ITA`), usando el índice (la fecha) como eje X. Seaborn detecta solo las columnas y les asigna un color distinto a cada una, además de agregar automáticamente una referencia (leyenda).
- `plt.title(...)`, `plt.xlabel(...)`, `plt.ylabel(...)`: igual que en matplotlib puro, ponen título y nombres de los ejes.
- `plt.show()`: muestra el gráfico en pantalla.

### Consigna

Mirando el gráfico:
1. ¿Qué selección se mantiene más tiempo en el puesto más bajo del gráfico (o sea, mejor ranking)? Brasil (BRA) es la que se mantiene más tiempo en el puesto más bajo del gráfico, sobre todo entre 1996 y 2000.
2. ¿Hay alguna selección que tenga un cambio brusco de ranking en algún período? Francia (FRA) muestra una caída pronunciada en su puesto de ranking a mediados de 1994, empeorando notablemente respecto a los meses anteriores.

---

## Cierre del laboratorio

Con `pd.read_csv` para cargar datos reales y `sns.lineplot` para graficarlos, ya podés tomar cualquier CSV con una columna de fechas y varias columnas numéricas, y visualizar su evolución en el tiempo.
