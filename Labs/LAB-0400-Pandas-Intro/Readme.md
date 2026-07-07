# LAB-0400 – Pandas: Introducción

## Objetivo

Conocer las dos estructuras básicas de pandas (`Series` y `DataFrame`), cómo crear datos (a mano o desde un archivo) y cómo explorar una tabla recién cargada.

## Requisitos previos

Tener instalado Python con la librería `pandas`. Si no la tenés instalada, ejecutar en una terminal:

```
pip install pandas
```

En todos los ejercicios hace falta importar pandas al principio del script:

```python
import pandas as pd
```

- `import pandas as pd`: importa la librería pandas y la renombra como `pd`, para escribir menos en cada línea.

---

## Ejercicio 1 – La clase `Series`

**¿Qué es?** Una `Series` es una columna de datos, parecida a una lista de Python, pero con un índice asociado a cada valor.

### Código

```python
import pandas as pd

edades = pd.Series([22, 25, 30, 45])

print(edades)
```

### Salida esperada

```
0    22
1    25
2    30
3    45
dtype: int64
```

### Explicación línea por línea

- `edades = pd.Series([22, 25, 30, 45])`: crea una `Series` a partir de una lista de números. `pd.Series(...)` es el constructor de la clase.
- `print(edades)`: muestra la Series. A la izquierda aparece el índice (0, 1, 2, 3), que pandas genera automáticamente empezando en 0. A la derecha están los valores.
- `dtype: int64`: indica el tipo de dato que contiene la Series (en este caso, números enteros).

### Consigna

Creá una Series con 5 nombres de alumnos (texto) y mostrala con `print()`. Fijate qué `dtype` aparece esta vez.

---

## Ejercicio 2 – La clase `DataFrame`

**¿Qué es?** Un `DataFrame` es una tabla completa, con filas y columnas. Es la estructura más usada en pandas. Se puede pensar como varias `Series` juntas, una por columna.

### Código

```python
import pandas as pd

datos = {
    "nombre": ["Juan", "María", "Pedro"],
    "edad": [22, 25, 30],
    "curso": ["Python", "Pandas", "Python"]
}

df = pd.DataFrame(datos)

print(df)
```

### Salida esperada

```
  nombre  edad   curso
0   Juan    22  Python
1  María    25  Pandas
2  Pedro    30  Python
```

### Explicación línea por línea

- `datos = {...}`: un diccionario de Python. Cada clave (`"nombre"`, `"edad"`, `"curso"`) va a ser una columna, y cada lista asociada son los valores de esa columna.
- `df = pd.DataFrame(datos)`: crea el `DataFrame` a partir del diccionario. `pd.DataFrame(...)` es el constructor de la clase.
- `print(df)`: muestra la tabla completa. A la izquierda aparece el índice de cada fila (0, 1, 2), generado automáticamente.

### Consigna

Agregá una cuarta persona al diccionario `datos` (agregando un valor a cada una de las 3 listas) y volvé a correr el código.

---

## Ejercicio 3 – Crear un DataFrame desde un diccionario (repaso ampliado)

Este ejercicio profundiza el `pd.DataFrame({...})` del ejercicio anterior, con un caso un poco más grande.

### Código

```python
import pandas as pd

ventas = {
    "producto": ["Notebook", "Mouse", "Teclado", "Monitor"],
    "precio":   [850000, 15000, 25000, 300000],
    "stock":    [5, 40, 20, 8]
}

df_ventas = pd.DataFrame(ventas)

print(df_ventas)
```

### Explicación línea por línea

- `ventas = {...}`: diccionario con 3 claves (`producto`, `precio`, `stock`), cada una con una lista de 4 valores.
- `df_ventas = pd.DataFrame(ventas)`: crea la tabla. Importante: todas las listas del diccionario tienen que tener la misma cantidad de elementos (acá, 4), porque cada posición forma una fila.
- `print(df_ventas)`: muestra la tabla.

### Consigna

¿Qué pasa si una de las listas del diccionario tiene menos elementos que las otras? Probá sacar un valor de la lista `stock` y volvé a correr el código para ver el error.

---

## Ejercicio 4 – Leer un archivo CSV

**¿Qué es un CSV?** Es un archivo de texto donde cada línea es una fila, y los valores de cada columna están separados por comas (o por otro carácter, como `;`). Se puede abrir con Excel, pero es texto plano.

### Código

```python
import pandas as pd

url = "https://raw.githubusercontent.com/estebancalabria/ml/refs/heads/main/datasets/fifa.csv"

df_fifa = pd.read_csv(url)

print(df_fifa.head())
```

### Explicación línea por línea

- `url = "https://..."`: guardamos en una variable la dirección del archivo, para no repetirla. En este caso el archivo está en internet, pero `pd.read_csv` funciona igual si le pasás la ruta de un archivo en tu computadora (por ejemplo, `"datos/fifa.csv"`).
- `df_fifa = pd.read_csv(url)`: lee el archivo CSV y lo convierte en un `DataFrame`. Pandas detecta solo dónde empieza cada columna, usando la coma como separador por defecto.
- `df_fifa.head()`: se explica en el Ejercicio 5, pero se usa acá para no imprimir la tabla entera si tiene muchas filas.

### Consigna

Cambiá `df_fifa.head()` por `print(df_fifa)` directamente (sin `.head()`) y notá la diferencia: ¿cuántas filas se imprimen ahora?

---

## Ejercicio 5 – Leer un archivo Excel

### Código

```python
import pandas as pd

df_excel = pd.read_excel("datos/ventas.xlsx")

print(df_excel.head())
```

### Explicación línea por línea

- `pd.read_excel("datos/ventas.xlsx")`: lee un archivo Excel (`.xlsx`) y lo convierte en un `DataFrame`. La sintaxis es igual a `pd.read_csv`, solo cambia el nombre de la función.
- Nota: para que esto funcione hace falta también instalar la librería `openpyxl`, con `pip install openpyxl`. Pandas la usa por dentro para leer archivos Excel.
- Si el archivo tiene más de una hoja, por defecto `pd.read_excel` lee la primera. Para leer una hoja específica se usa el parámetro `sheet_name`, por ejemplo: `pd.read_excel("datos/ventas.xlsx", sheet_name="Enero")`.

### Consigna

Si tenés un archivo Excel propio a mano, probá cargarlo cambiando la ruta del archivo. Si no tenés uno, dejá este ejercicio como lectura y seguí al próximo.

---

## Ejercicio 6 – Ver las primeras y últimas filas: `head()` y `tail()`

Volvemos al DataFrame de ventas del Ejercicio 3 para explorar sus datos.

### Código

```python
import pandas as pd

ventas = {
    "producto": ["Notebook", "Mouse", "Teclado", "Monitor", "Auriculares", "Webcam"],
    "precio":   [850000, 15000, 25000, 300000, 45000, 60000],
    "stock":    [5, 40, 20, 8, 15, 10]
}

df_ventas = pd.DataFrame(ventas)

print(df_ventas.head(3))
print(df_ventas.tail(2))
```

### Explicación línea por línea

- `df_ventas.head(3)`: devuelve las primeras 3 filas de la tabla. Si no le pasamos un número (`df_ventas.head()`), muestra las primeras 5 por defecto.
- `df_ventas.tail(2)`: devuelve las últimas 2 filas de la tabla. Igual que `head()`, sin argumento muestra las últimas 5 por defecto.
- Estos métodos son útiles cuando la tabla tiene muchas filas y no queremos imprimirlas todas para revisar que los datos se cargaron bien.

### Consigna

Probá `df_ventas.head(1)` y `df_ventas.tail(1)`. ¿Qué fila devuelve cada uno?

---

## Ejercicio 7 – Información general: `info()`

### Código

```python
df_ventas.info()
```

### Salida esperada

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 6 entries, 0 to 5
Data columns (total 3 columns):
 #   Column    Non-Null Count  Dtype
---  ------    --------------  -----
 0   producto  6 non-null      object
 1   precio    6 non-null      int64
 2   stock     6 non-null      int64
dtypes: int64(2), object(1)
memory usage: 272.0+ bytes
```

### Explicación línea por línea

- `df_ventas.info()`: no recibe argumentos, y no hace falta usar `print()` porque ya imprime el resultado por su cuenta.
- La salida muestra, por cada columna: su nombre, cuántos valores no nulos tiene (`Non-Null Count`) y su tipo de dato (`Dtype`).
- `object` es el tipo que usa pandas para texto (strings). `int64` es un número entero.
- Esto sirve para detectar rápido si hay columnas con datos faltantes (si `Non-Null Count` es menor a la cantidad total de filas) y si el tipo de cada columna es el esperado (por ejemplo, si una columna de números quedó cargada como texto).

### Consigna

Ejecutá `df_fifa.info()` (del Ejercicio 4) y compará: ¿tiene la misma cantidad de columnas que viste con `head()`?

---

## Ejercicio 8 – Estadísticas básicas: `describe()`

### Código

```python
print(df_ventas.describe())
```

### Salida esperada

```
             precio      stock
count  6.000000e+00   6.000000
mean   2.158333e+05  16.333333
std    3.146650e+05  12.421356
min    1.500000e+04   5.000000
25%    2.750000e+04   8.750000
50%    5.250000e+04  12.500000
75%    2.375000e+05  18.750000
max    8.500000e+05  40.000000
```

### Explicación línea por línea

- `df_ventas.describe()`: calcula estadísticas básicas de todas las columnas numéricas de la tabla (acá, `precio` y `stock`). La columna `producto` no aparece porque es texto, no números.
- `count`: cantidad de valores no nulos.
- `mean`: promedio.
- `std`: desvío estándar (qué tan dispersos están los valores respecto al promedio).
- `min` / `max`: valor mínimo y máximo.
- `25%`, `50%`, `75%`: percentiles (por ejemplo, el 50% es la mediana, el valor que deja la mitad de los datos por debajo y la mitad por arriba).

### Consigna

Mirando la salida, ¿cuál es el precio promedio de los productos? ¿Cuál es el producto con el precio más alto (podés compararlo con la tabla original)?

---

## Ejercicio 9 – Forma de la tabla: `shape`

### Código

```python
print(df_ventas.shape)
```

### Salida esperada

```
(6, 3)
```

### Explicación línea por línea

- `df_ventas.shape`: no es un método (no lleva paréntesis), sino un atributo. Devuelve una tupla con dos números: la cantidad de filas y la cantidad de columnas.
- En este caso, `(6, 3)` significa 6 filas y 3 columnas.

### Consigna

¿Cuántas filas y columnas esperás que tenga `df_fifa.shape` (del Ejercicio 4), según lo que viste con `head()`? Comprobalo ejecutando el código.

---

## Ejercicio 10 – Nombres de columnas: `columns`

### Código

```python
print(df_ventas.columns)
```

### Salida esperada

```
Index(['producto', 'precio', 'stock'], dtype='object')
```

### Explicación línea por línea

- `df_ventas.columns`: es otro atributo (sin paréntesis). Devuelve los nombres de todas las columnas de la tabla.
- Sirve para saber rápido cómo se llama cada columna, sobre todo cuando la tabla viene de un archivo externo y no sabemos de memoria los nombres exactos.

### Consigna

Ejecutá `df_fifa.columns` (del Ejercicio 4). ¿Coincide con lo que habías visto en el `head()`?

---

## Ejercicio 11 – Tipo de dato de cada columna: `dtypes`

### Código

```python
print(df_ventas.dtypes)
```

### Salida esperada

```
producto    object
precio       int64
stock        int64
dtype: object
```

### Explicación línea por línea

- `df_ventas.dtypes`: atributo (sin paréntesis) que devuelve el tipo de dato de cada columna, una por una.
- Es parecido a una parte de lo que ya vimos en `info()`, pero acá solo se muestran los tipos, sin la demás información.

### Consigna

Compará la salida de `df_ventas.dtypes` con la de `df_ventas.info()` del Ejercicio 7. ¿Qué información aparece en `info()` que no está en `dtypes`?

---

## Cierre del laboratorio

Con `Series` y `DataFrame` ya conocés las dos estructuras principales de pandas. Con `pd.DataFrame({...})`, `pd.read_csv()` y `pd.read_excel()` sabés crear o cargar datos. Y con `head()`, `tail()`, `info()`, `describe()`, `shape`, `columns` y `dtypes` ya tenés las herramientas básicas para revisar cualquier tabla apenas la cargás, antes de empezar a trabajar con ella.
