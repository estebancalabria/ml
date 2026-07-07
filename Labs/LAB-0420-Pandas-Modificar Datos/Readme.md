# LAB-0420 – Pandas: Modificar Datos

## Objetivo

Aprender a modificar un DataFrame: crear o sobrescribir columnas, eliminar columnas, renombrar columnas, manejar valores nulos y convertir tipos de dato.

## Requisitos previos

Tener instalado Python con la librería `pandas`.

```
pip install pandas
```

## El dataset

Vamos a usar el mismo DataFrame de ventas a lo largo del laboratorio, creado en memoria:

```python
import pandas as pd
import numpy as np

ventas = {
    "producto": ["Notebook", "Mouse", "Teclado", "Monitor", "Auriculares", "Webcam"],
    "precio":   [850000, 15000, 25000, 300000, 45000, 60000],
    "stock":    [5, 40, 20, np.nan, 15, 10]
}

df = pd.DataFrame(ventas)

print(df)
```

### Salida

```
      producto    precio  stock
0     Notebook  850000.0    5.0
1        Mouse   15000.0   40.0
2      Teclado   25000.0   20.0
3      Monitor  300000.0    NaN
4  Auriculares   45000.0   15.0
5       Webcam   60000.0   10.0
```

- `np.nan` es la forma en que numpy representa un valor faltante (nulo). Lo agregamos a propósito en la fila de Monitor para poder practicar el manejo de nulos más adelante.
- Notá que, al haber un `NaN` en la columna `stock`, pandas convirtió toda la columna a decimales (`float`), aunque el resto de los valores sean enteros. Esto es normal: una columna no puede mezclar enteros con `NaN` en la misma columna.

---

## Ejercicio 1 – Crear una columna nueva: `df["nueva_col"] = ...`

### Código

```python
df["precio_con_iva"] = df["precio"] * 1.21

print(df)
```

### Salida

```
      producto    precio  stock  precio_con_iva
0     Notebook  850000.0    5.0        1028500.0
1        Mouse   15000.0   40.0          18150.0
2      Teclado   25000.0   20.0          30250.0
3      Monitor  300000.0    NaN         363000.0
4  Auriculares   45000.0   15.0          54450.0
5       Webcam   60000.0   10.0          72600.0
```

### Explicación línea por línea

- `df["precio_con_iva"] = df["precio"] * 1.21`: si la columna `"precio_con_iva"` no existe en el DataFrame, pandas la crea. El valor que le asignamos es el resultado de multiplicar, fila por fila, la columna `precio` por 1.21.
- Esta misma línea, si `"precio_con_iva"` ya existiera, sobrescribiría sus valores en vez de crear una columna nueva.

### Consigna

Creá una columna nueva llamada `stock_bajo` que valga `True` cuando `stock` sea menor a 15, y `False` en caso contrario. Pista: `df["stock"] < 15`.

---

## Ejercicio 2 – Eliminar columnas: `df.drop(columns=["col"])`

### Código

```python
df_sin_iva = df.drop(columns=["precio_con_iva"])

print(df_sin_iva)
```

### Salida

```
      producto    precio  stock
0     Notebook  850000.0    5.0
1        Mouse   15000.0   40.0
2      Teclado   25000.0   20.0
3      Monitor  300000.0    NaN
4  Auriculares   45000.0   15.0
5       Webcam   60000.0   10.0
```

### Explicación línea por línea

- `df.drop(columns=["precio_con_iva"])`: elimina la columna indicada. El argumento `columns` recibe una lista, así que se pueden eliminar varias a la vez, por ejemplo `columns=["precio_con_iva", "stock"]`.
- Importante: `df.drop(...)` devuelve un DataFrame nuevo con la columna eliminada, pero no modifica `df` original. Por eso lo guardamos en una variable nueva (`df_sin_iva`). Si quisiéramos modificar `df` directamente, habría que escribir `df = df.drop(columns=["precio_con_iva"])`.

### Consigna

Eliminá la columna `stock_bajo` que creaste en el Ejercicio 1, guardando el resultado en `df` (o sea, sobrescribiendo la variable original).

---

## Ejercicio 3 – Renombrar columnas: `df.rename(columns={"viejo": "nuevo"})`

### Código

```python
df_renombrado = df.rename(columns={"producto": "nombre_producto", "precio": "precio_ars"})

print(df_renombrado)
```

### Salida

```
  nombre_producto  precio_ars  stock
0        Notebook    850000.0    5.0
1           Mouse     15000.0   40.0
2         Teclado     25000.0   20.0
3         Monitor    300000.0    NaN
4     Auriculares     45000.0   15.0
5          Webcam     60000.0   10.0
```

### Explicación línea por línea

- `df.rename(columns={"producto": "nombre_producto", "precio": "precio_ars"})`: el argumento `columns` recibe un diccionario, donde cada clave es el nombre viejo y cada valor es el nombre nuevo. Se pueden renombrar una o varias columnas en la misma llamada.
- Igual que `drop`, `rename` devuelve un DataFrame nuevo por defecto, sin modificar el original.

### Consigna

Renombrá la columna `stock` a `unidades_disponibles`, guardando el resultado en una variable nueva llamada `df_stock_renombrado`.

---

## Ejercicio 4 – Reemplazar valores nulos: `df.fillna(valor)`

### Código

```python
df_completo = df.fillna(0)

print(df_completo)
```

### Salida

```
      producto    precio  stock
0     Notebook  850000.0    5.0
1        Mouse   15000.0   40.0
2      Teclado   25000.0   20.0
3      Monitor  300000.0    0.0
4  Auriculares   45000.0   15.0
5       Webcam   60000.0   10.0
```

### Explicación línea por línea

- `df.fillna(0)`: reemplaza todos los valores nulos (`NaN`) del DataFrame por el valor indicado, en este caso `0`. La fila de Monitor, que tenía `NaN` en `stock`, ahora tiene `0.0`.
- Se puede aplicar sobre una sola columna en vez de todo el DataFrame, por ejemplo: `df["stock"] = df["stock"].fillna(0)`.
- El valor de relleno no tiene que ser siempre `0`: también es común usar el promedio de la columna, por ejemplo `df["stock"].fillna(df["stock"].mean())`.

### Consigna

Reemplazá el valor nulo de `stock` usando el promedio de esa columna en vez de `0`. Pista: `df["stock"].fillna(df["stock"].mean())`.

---

## Ejercicio 5 – Eliminar filas con valores nulos: `df.dropna()`

### Código

```python
df_sin_nulos = df.dropna()

print(df_sin_nulos)
```

### Salida

```
      producto    precio  stock
0     Notebook  850000.0    5.0
1        Mouse   15000.0   40.0
2      Teclado   25000.0   20.0
4  Auriculares   45000.0   15.0
5       Webcam   60000.0   10.0
```

### Explicación línea por línea

- `df.dropna()`: elimina cualquier fila que tenga al menos un valor nulo en alguna de sus columnas. En este caso, la fila de Monitor (índice 3) desaparece porque tenía `NaN` en `stock`.
- A diferencia de `fillna`, acá no rellenamos el dato faltante, directamente sacamos la fila entera de la tabla.
- Se puede limitar a columnas específicas con el argumento `subset`, por ejemplo `df.dropna(subset=["stock"])`, para que solo elimine filas si el nulo está en esa columna puntual.

### Consigna

Contá cuántas filas tiene `df_sin_nulos` con `df_sin_nulos.shape`, y compará con las filas de `df` original.

---

## Ejercicio 6 – Convertir el tipo de dato de una columna: `df.astype(tipo)`

### Código

```python
df_relleno = df.fillna(0)
df_relleno["stock"] = df_relleno["stock"].astype(int)

print(df_relleno)
print(df_relleno.dtypes)
```

### Salida

```
      producto    precio  stock
0     Notebook  850000.0      5
1        Mouse   15000.0     40
2      Teclado   25000.0     20
3      Monitor  300000.0      0
4  Auriculares   45000.0     15
5       Webcam   60000.0     10

producto     object
precio      float64
stock         int64
dtype: object
```

### Explicación línea por línea

- `df_relleno = df.fillna(0)`: primero rellenamos los nulos, porque no se puede convertir una columna con `NaN` a números enteros (`int`) directamente.
- `df_relleno["stock"] = df_relleno["stock"].astype(int)`: convierte la columna `stock` (que estaba en `float`, con decimales como `5.0`) a `int` (números enteros, `5`). El resultado se vuelve a guardar en la misma columna, sobrescribiéndola.
- `df_relleno.dtypes`: lo usamos para confirmar que `stock` ahora es `int64` en vez de `float64`.

### Consigna

Convertí la columna `precio` a tipo `int` (en vez de `float`) y confirmá el cambio con `.dtypes`.

---

## Cierre del laboratorio

Con `df["nueva_col"] = ...` creás o sobrescribís columnas. Con `drop` las eliminás y con `rename` las renombrás. Con `fillna` y `dropna` manejás valores nulos, ya sea rellenándolos o eliminando esas filas. Y con `astype` controlás el tipo de dato de cada columna, algo importante para que los cálculos y gráficos posteriores funcionen como se espera.
