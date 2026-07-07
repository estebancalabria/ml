# LAB-0430 – Pandas: Agrupar y Agregar

## Objetivo

Aprender a agrupar filas con `groupby`, aplicar funciones de agregación (`mean`, `sum`, `count`, `agg`), ordenar filas con `sort_values` y contar valores únicos con `value_counts`.

## Requisitos previos

Tener instalado Python con la librería `pandas`.

```
pip install pandas
```

## El dataset

Vamos a usar un DataFrame de ventas con una columna de categoría, creado en memoria:

```python
import pandas as pd

ventas = {
    "producto":  ["Notebook", "Mouse", "Teclado", "Monitor", "Auriculares", "Webcam", "Mouse Gamer", "Monitor 4K"],
    "categoria": ["Computación", "Accesorios", "Accesorios", "Computación", "Accesorios", "Accesorios", "Accesorios", "Computación"],
    "precio":    [850000, 15000, 25000, 300000, 45000, 60000, 35000, 500000],
    "stock":     [5, 40, 20, 8, 15, 10, 12, 3]
}

df = pd.DataFrame(ventas)

print(df)
```

### Salida

```
      producto    categoria  precio  stock
0     Notebook  Computación  850000      5
1        Mouse   Accesorios   15000     40
2      Teclado   Accesorios   25000     20
3      Monitor  Computación  300000      8
4  Auriculares   Accesorios   45000     15
5       Webcam   Accesorios   60000     10
6  Mouse Gamer   Accesorios   35000     12
7   Monitor 4K  Computación  500000      3
```

---

## Ejercicio 1 – Agrupar filas: `df.groupby("columna")`

### Código

```python
grupos = df.groupby("categoria")

print(grupos)
```

### Salida

```
<pandas.core.groupby.generic.DataFrameGroupBy object at 0x...>
```

### Explicación línea por línea

- `df.groupby("categoria")`: agrupa las filas de `df` según los valores de la columna `categoria`. En este dataset hay dos grupos: `"Computación"` y `"Accesorios"`.
- `grupos` no es una tabla, es un objeto especial de pandas que representa "las filas ya separadas en grupos, pero todavía sin calcular nada sobre ellas". Por eso, al imprimirlo, no se ve una tabla sino una referencia al objeto.
- `groupby` casi nunca se usa solo: se combina con una función de agregación (como se ve en los ejercicios siguientes) para obtener un resultado útil.

### Consigna

Agrupá el DataFrame por la columna `producto` en vez de `categoria`, y guardalo en una variable `grupos_producto`. ¿Cuántos grupos esperás que se formen, dado que cada producto es distinto?

---

## Ejercicio 2 – Promedio por grupo: `.mean()`

### Código

```python
promedio_por_categoria = df.groupby("categoria")[["precio", "stock"]].mean()

print(promedio_por_categoria)
```

### Salida

```
                   precio      stock
categoria
Accesorios    36000.000000  19.400000
Computación  550000.000000   5.333333
```

### Explicación línea por línea

- `df.groupby("categoria")`: agrupa las filas por categoría, como en el ejercicio anterior.
- `[["precio", "stock"]]`: de todas las columnas numéricas disponibles, seleccionamos solo estas dos para calcular el promedio (si no las filtráramos, pandas intentaría promediar también columnas de texto, lo cual daría error o se ignoraría según la versión).
- `.mean()`: calcula el promedio de esas columnas, separado por cada grupo.
- El resultado es un DataFrame donde el índice ya no es 0, 1, 2..., sino los nombres de cada categoría.

### Consigna

Calculá el promedio de `precio` y `stock`, pero agrupando por `producto` en vez de `categoria`. ¿Tiene sentido el resultado, dado que cada producto aparece una sola vez?

---

## Ejercicio 3 – Suma por grupo: `.sum()`

### Código

```python
suma_por_categoria = df.groupby("categoria")[["stock"]].sum()

print(suma_por_categoria)
```

### Salida

```
             stock
categoria
Accesorios      97
Computación     16
```

### Explicación línea por línea

- `df.groupby("categoria")[["stock"]].sum()`: agrupa por categoría y suma los valores de `stock` dentro de cada grupo.
- Esto responde la pregunta "¿cuántas unidades hay en stock, en total, por cada categoría?".
- La estructura de la línea es la misma que en el Ejercicio 2, solo cambia la función de agregación al final (`.sum()` en vez de `.mean()`).

### Consigna

Calculá la suma de `precio` por categoría. ¿Qué categoría tiene el valor total más alto?

---

## Ejercicio 4 – Cantidad de filas por grupo: `.count()`

### Código

```python
cantidad_por_categoria = df.groupby("categoria")[["producto"]].count()

print(cantidad_por_categoria)
```

### Salida

```
             producto
categoria
Accesorios          5
Computación          3
```

### Explicación línea por línea

- `df.groupby("categoria")[["producto"]].count()`: agrupa por categoría y cuenta cuántos valores no nulos hay en la columna `producto`, dentro de cada grupo.
- En la práctica, esto equivale a contar cuántas filas (productos) hay en cada categoría, ya que la columna `producto` no tiene valores nulos.
- `count()` cuenta valores no nulos, a diferencia de `size()` (que no vimos acá) que cuenta filas sin importar si hay nulos o no.

### Consigna

¿Cuántos productos hay en la categoría `"Computación"`, según el resultado?

---

## Ejercicio 5 – Varias agregaciones a la vez: `.agg(...)`

### Código

```python
resumen = df.groupby("categoria")[["precio", "stock"]].agg(["mean", "sum"])

print(resumen)
```

### Salida

```
                   precio                stock
                     mean       sum       mean sum
categoria
Accesorios    36000.000000    180000  19.400000   97
Computación  550000.000000   1650000   5.333333   16
```

### Explicación línea por línea

- `.agg(["mean", "sum"])`: en vez de aplicar una sola función de agregación, `agg` permite pasar una lista con varias funciones a la vez (acá, promedio y suma). El resultado calcula ambas para cada columna seleccionada (`precio` y `stock`).
- Esto evita tener que escribir una línea separada por cada función, como hicimos en los ejercicios 2 y 3.
- También se puede usar `agg` con un diccionario, para aplicar una función distinta a cada columna, por ejemplo: `df.groupby("categoria").agg({"precio": "mean", "stock": "sum"})`.

### Consigna

Probá la versión con diccionario del párrafo anterior (`{"precio": "mean", "stock": "sum"}`) y compará el resultado con el de la tabla del ejercicio.

---

## Ejercicio 6 – Ordenar filas: `df.sort_values("columna")`

### Código

```python
ordenado_por_precio = df.sort_values("precio")

print(ordenado_por_precio)
```

### Salida

```
      producto    categoria  precio  stock
1        Mouse   Accesorios   15000     40
2      Teclado   Accesorios   25000     20
6  Mouse Gamer   Accesorios   35000     12
4  Auriculares   Accesorios   45000     15
5       Webcam   Accesorios   60000     10
3      Monitor  Computación  300000      8
7   Monitor 4K  Computación  500000      3
0     Notebook  Computación  850000      5
```

### Explicación línea por línea

- `df.sort_values("precio")`: ordena todas las filas del DataFrame según los valores de la columna `precio`, de menor a mayor por defecto.
- Notá que el índice original (los números de la izquierda) se mantiene asociado a cada fila, aunque el orden visual haya cambiado. Por eso el índice ya no aparece en orden (1, 2, 6, 4, 5, 3, 7, 0).
- Para ordenar de mayor a menor, se agrega el argumento `ascending=False`: `df.sort_values("precio", ascending=False)`.

### Consigna

Ordená el DataFrame por `stock`, de mayor a menor. ¿Qué producto queda primero?

---

## Ejercicio 7 – Contar valores únicos: `df.value_counts()`

### Código

```python
conteo_categorias = df["categoria"].value_counts()

print(conteo_categorias)
```

### Salida

```
categoria
Accesorios     5
Computación    3
Name: count, dtype: int64
```

### Explicación línea por línea

- `df["categoria"]`: primero seleccionamos la columna sobre la que queremos contar valores (una `Series`, no el DataFrame completo).
- `.value_counts()`: cuenta cuántas veces aparece cada valor distinto en esa columna, y ordena el resultado de mayor a menor cantidad por defecto.
- El resultado es parecido al del Ejercicio 4 (`count()` con `groupby`), pero `value_counts()` es más directo cuando solo queremos contar frecuencias de una sola columna, sin necesidad de armar un `groupby`.

### Consigna

Usá `value_counts()` sobre la columna `producto`. ¿Por qué cada valor aparece exactamente una vez?

---

## Cierre del laboratorio

Con `groupby` agrupás filas según una columna, y con `.mean()`, `.sum()`, `.count()` o `.agg(...)` calculás resultados sobre esos grupos. Con `sort_values` ordenás las filas de una tabla según una columna, y con `value_counts()` contás rápidamente cuántas veces aparece cada valor en una columna. Estas herramientas son las que más se usan para resumir y entender un dataset antes de graficarlo o modelarlo.
