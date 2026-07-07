# LAB-0410 – Pandas: Seleccionar Datos

## Objetivo

Aprender las distintas formas de seleccionar columnas, filas y subconjuntos de un DataFrame: por nombre de columna, por etiqueta (`loc`), por posición (`iloc`) y por condición (filtrado).

## Requisitos previos

Tener instalado Python con la librería `pandas`.

```
pip install pandas
```

## El dataset

Vamos a usar el mismo DataFrame de ventas a lo largo de todo el laboratorio, creado en memoria:

```python
import pandas as pd

ventas = {
    "producto": ["Notebook", "Mouse", "Teclado", "Monitor", "Auriculares", "Webcam"],
    "precio":   [850000, 15000, 25000, 300000, 45000, 60000],
    "stock":    [5, 40, 20, 8, 15, 10]
}

df = pd.DataFrame(ventas)

print(df)
```

### Salida

```
      producto  precio  stock
0     Notebook  850000      5
1        Mouse   15000     40
2      Teclado   25000     20
3      Monitor  300000      8
4  Auriculares   45000     15
5       Webcam   60000     10
```

---

## Ejercicio 1 – Seleccionar una columna: `df["columna"]`

### Código

```python
precios = df["precio"]
print(precios)
```

### Salida

```
0    850000
1     15000
2     25000
3    300000
4     45000
5     60000
Name: precio, dtype: int64
```

### Explicación línea por línea

- `df["precio"]`: selecciona la columna `precio` usando su nombre entre corchetes. El resultado es una `Series` (una sola columna con su índice), no un `DataFrame`.
- `Name: precio`: pandas muestra el nombre de la columna al final, para dejar en claro de qué Series se trata.

### Consigna

Seleccioná la columna `producto` y guardala en una variable llamada `productos`. Imprimila.

---

## Ejercicio 2 – Seleccionar varias columnas: `df[["col1", "col2"]]`

### Código

```python
subset = df[["producto", "precio"]]
print(subset)
```

### Salida

```
      producto  precio
0     Notebook  850000
1        Mouse   15000
2      Teclado   25000
3      Monitor  300000
4  Auriculares   45000
5       Webcam   60000
```

### Explicación línea por línea

- `df[["producto", "precio"]]`: hay dos pares de corchetes. El corchete de afuera es la selección sobre `df`. El corchete de adentro (`["producto", "precio"]`) es una lista con los nombres de las columnas que queremos.
- A diferencia del Ejercicio 1, acá el resultado es un `DataFrame` (una tabla), porque pedimos más de una columna.

### Consigna

Seleccioná las columnas `producto` y `stock` juntas, en ese orden, y comprobá que el resultado tiene 2 columnas.

---

## Ejercicio 3 – Selección por etiqueta: `df.loc[fila, columna]`

**¿Qué es una etiqueta?** Es el nombre con el que se identifica una fila o columna, no su posición numérica. En nuestro DataFrame, las etiquetas de fila son 0, 1, 2, 3, 4, 5 (el índice) y las etiquetas de columna son `"producto"`, `"precio"`, `"stock"`.

### Código

```python
# Un valor puntual: fila con etiqueta 2, columna "producto"
valor = df.loc[2, "producto"]
print(valor)
```

### Salida

```
Teclado
```

### Código (varias filas y columnas)

```python
subset = df.loc[0:2, ["producto", "precio"]]
print(subset)
```

### Salida

```
   producto  precio
0  Notebook  850000
1     Mouse   15000
2   Teclado   25000
```

### Explicación línea por línea

- `df.loc[2, "producto"]`: primero se indica la fila (etiqueta `2`), después la columna (etiqueta `"producto"`), separadas por coma. Devuelve el valor puntual de esa celda.
- `df.loc[0:2, ["producto", "precio"]]`: `0:2` es un rango de etiquetas de fila, desde la 0 hasta la 2. Importante: con `loc`, el rango incluye el final (`2` queda incluido), a diferencia de un rango normal de Python.
- `["producto", "precio"]`: lista de columnas a mostrar.

### Consigna

Usá `df.loc` para obtener el valor de `stock` de la fila con etiqueta 4 (Auriculares).

---

## Ejercicio 4 – Selección por posición: `df.iloc[fila, columna]`

**¿Qué es una posición?** Es el número de orden, empezando en 0, sin importar cómo se llame la etiqueta. En este DataFrame coincide con la etiqueta del índice, pero no siempre es así (por ejemplo, si el índice fueran fechas o nombres).

### Código

```python
# Un valor puntual: fila en posición 2, columna en posición 0
valor = df.iloc[2, 0]
print(valor)
```

### Salida

```
Teclado
```

### Código (varias filas y columnas)

```python
subset = df.iloc[0:2, 0:2]
print(subset)
```

### Salida

```
   producto  precio
0  Notebook  850000
1     Mouse   15000
```

### Explicación línea por línea

- `df.iloc[2, 0]`: fila en la posición 2 (la tercera fila, contando desde 0), columna en la posición 0 (la primera columna). Devuelve `"Teclado"`, igual que en el ejercicio anterior, pero acá usamos números en vez de nombres.
- `df.iloc[0:2, 0:2]`: rango de posiciones de fila (0 y 1) y de columna (0 y 1). A diferencia de `loc`, acá el rango NO incluye el final: `0:2` trae las posiciones 0 y 1, pero no la 2.

### Consigna

Usá `df.iloc` con números para obtener el mismo valor de `stock` de Auriculares que obtuviste en el Ejercicio 3 con `loc`. ¿Qué números de fila y columna usaste?

---

## Ejercicio 5 – Filtrado por condición: `df[df["columna"] > valor]`

### Código

```python
caros = df[df["precio"] > 50000]
print(caros)
```

### Salida

```
   producto  precio  stock
0  Notebook  850000      5
3   Monitor  300000      8
5    Webcam   60000     10
```

### Explicación línea por línea

- `df["precio"] > 50000`: esta parte, evaluada sola, devuelve una Series de valores `True`/`False`, uno por cada fila, según si el precio de esa fila es mayor a 50000.
- `df[...]`: al poner esa Series de `True`/`False` dentro de los corchetes de `df`, pandas devuelve solamente las filas donde el valor fue `True`.
- El resultado son las filas de Notebook, Monitor y Webcam, que son los productos con precio mayor a 50000.

### Consigna

Filtrá el DataFrame para quedarte solo con los productos que tengan `stock` menor a 15. ¿Cuántas filas quedan?

---

## Cierre del laboratorio

Con `df["columna"]` y `df[["col1", "col2"]]` seleccionás columnas por nombre. Con `df.loc` seleccionás por etiqueta (nombre de fila/columna) y con `df.iloc` por posición numérica. Con `df[condición]` filtrás filas según un criterio. Estas cinco formas cubren la gran mayoría de los casos de selección de datos en pandas.
