# LAB-0440 – Pandas: Combinar Tablas e Índices

## Objetivo

Aprender a unir dos tablas con `merge`, apilarlas con `concat`, y manejar el índice de un DataFrame con `set_index` y `reset_index`.

## Requisitos previos

Tener instalado Python con la librería `pandas`.

```
pip install pandas
```

## El dataset

Vamos a usar dos DataFrames chicos creados en memoria: uno de productos y otro de categorías.

```python
import pandas as pd

productos = pd.DataFrame({
    "id_producto": [1, 2, 3, 4],
    "nombre": ["Notebook", "Mouse", "Teclado", "Monitor"],
    "id_categoria": [10, 20, 20, 10]
})

categorias = pd.DataFrame({
    "id_categoria": [10, 20],
    "nombre_categoria": ["Computación", "Accesorios"]
})

print(productos)
print(categorias)
```

### Salida

```
   id_producto    nombre  id_categoria
0            1  Notebook            10
1            2     Mouse            20
2            3   Teclado            20
3            4   Monitor            10

   id_categoria nombre_categoria
0            10      Computación
1            20       Accesorios
```

---

## Ejercicio 1 – Unir dos tablas: `pd.merge(df1, df2, on="columna")`

### Código

```python
productos_completos = pd.merge(productos, categorias, on="id_categoria")

print(productos_completos)
```

### Salida

```
   id_producto    nombre  id_categoria nombre_categoria
0            1  Notebook            10      Computación
1            2     Mouse            20       Accesorios
2            3   Teclado            20       Accesorios
3            4   Monitor            10      Computación
```

### Explicación línea por línea

- `pd.merge(productos, categorias, on="id_categoria")`: junta las dos tablas en una sola, usando la columna `id_categoria` como referencia común. Es igual a un JOIN de SQL: por cada fila de `productos`, busca la fila de `categorias` que tenga el mismo `id_categoria`, y agrega sus columnas.
- `on="id_categoria"`: indica cuál es la columna en común entre las dos tablas. Tiene que existir con el mismo nombre en ambas.
- El resultado tiene las columnas de `productos` más las columnas de `categorias` (menos la columna `id_categoria`, que queda una sola vez).
- Por defecto, `pd.merge` hace un `inner join`: solo mantiene las filas que tienen coincidencia en ambas tablas. Si un producto tuviera un `id_categoria` que no existe en `categorias`, esa fila se perdería.

### Consigna

Agregá una fila nueva a `productos` con un `id_categoria` que no exista en `categorias` (por ejemplo, `30`) y volvé a correr el `merge`. ¿Qué pasa con esa fila en el resultado?

---

## Ejercicio 2 – Apilar tablas por filas: `pd.concat([df1, df2])`

### Código

```python
ventas_enero = pd.DataFrame({
    "producto": ["Notebook", "Mouse"],
    "cantidad": [3, 10]
})

ventas_febrero = pd.DataFrame({
    "producto": ["Teclado", "Monitor"],
    "cantidad": [5, 2]
})

ventas_totales = pd.concat([ventas_enero, ventas_febrero])

print(ventas_totales)
```

### Salida

```
   producto  cantidad
0  Notebook         3
1     Mouse        10
0   Teclado         5
1   Monitor         2
```

### Explicación línea por línea

- `pd.concat([ventas_enero, ventas_febrero])`: apila las filas de las dos tablas, una debajo de la otra. Por defecto apila por filas (`axis=0`).
- Notá que el índice se repite (0, 1, 0, 1), porque cada tabla original tenía su propio índice empezando en 0, y `concat` no lo reordena solo.
- Para corregir el índice repetido, se agrega el argumento `ignore_index=True`: `pd.concat([ventas_enero, ventas_febrero], ignore_index=True)`, que genera un índice nuevo y continuo (0, 1, 2, 3).

### Consigna

Volvé a correr el `concat` agregando `ignore_index=True` y comparalo con el resultado de arriba. ¿Cómo queda el índice ahora?

---

## Ejercicio 3 – Apilar tablas por columnas: `pd.concat([df1, df2], axis=1)`

### Código

```python
nombres = pd.DataFrame({"nombre": ["Notebook", "Mouse", "Teclado"]})
precios = pd.DataFrame({"precio": [850000, 15000, 25000]})

productos_juntos = pd.concat([nombres, precios], axis=1)

print(productos_juntos)
```

### Salida

```
     nombre  precio
0  Notebook  850000
1     Mouse   15000
2   Teclado   25000
```

### Explicación línea por línea

- `axis=1`: le indica a `concat` que en vez de apilar filas, tiene que pegar las tablas una al lado de la otra, como columnas nuevas.
- Esto funciona bien cuando ambas tablas tienen la misma cantidad de filas y el mismo orden, porque `concat` las junta por posición (fila 0 con fila 0, fila 1 con fila 1, etc.), no por un valor en común como hace `merge`.

### Consigna

¿Qué diferencia le encontrás a `pd.concat([...], axis=1)` respecto a `pd.merge(...)` del Ejercicio 1? Pensá en qué pasaría si las filas de `nombres` y `precios` estuvieran en distinto orden.

---

## Ejercicio 4 – Usar una columna como índice: `df.set_index("columna")`

### Código

```python
productos_indexado = productos.set_index("id_producto")

print(productos_indexado)
```

### Salida

```
                nombre  id_categoria
id_producto
1             Notebook            10
2                Mouse            20
3              Teclado            20
4              Monitor            10
```

### Explicación línea por línea

- `productos.set_index("id_producto")`: convierte la columna `id_producto` en el índice del DataFrame, en vez de dejarla como una columna más. El índice por defecto (0, 1, 2, 3) desaparece y se reemplaza por los valores de `id_producto`.
- Esto es útil cuando una columna identifica de forma única a cada fila (como un ID), porque después se puede acceder directamente a una fila con `df.loc[id]`, sin tener que filtrar por condición.
- Igual que `drop` y `rename`, `set_index` devuelve un DataFrame nuevo por defecto, sin modificar el original.

### Consigna

Con `productos_indexado`, usá `.loc[3]` para obtener directamente la fila del producto con `id_producto` igual a 3. Compará esto con cómo tendrías que hacerlo si `id_producto` siguiera siendo una columna común (filtrando con `df[df["id_producto"] == 3]`).

---

## Ejercicio 5 – Volver el índice a columna: `df.reset_index()`

### Código

```python
productos_vuelto = productos_indexado.reset_index()

print(productos_vuelto)
```

### Salida

```
   id_producto    nombre  id_categoria
0            1  Notebook            10
1            2     Mouse            20
2            3   Teclado            20
3            4   Monitor            10
```

### Explicación línea por línea

- `productos_indexado.reset_index()`: hace el proceso inverso al `set_index` del ejercicio anterior. Toma el índice actual (`id_producto`) y lo vuelve a convertir en una columna normal, mientras que el índice vuelve a ser el genérico (0, 1, 2, 3).
- Esto es útil, por ejemplo, después de un `groupby` (donde la columna de agrupación queda como índice) cuando querés volver a tener esa columna disponible para filtrar o graficar como cualquier otra.

### Consigna

Agrupá `productos` por `id_categoria` usando `groupby` y contá cuántos productos hay por categoría (como en el laboratorio anterior). Después aplicá `reset_index()` al resultado. ¿Qué cambia en la tabla?

---

## Cierre del laboratorio

Con `pd.merge` combinás dos tablas relacionadas por una columna en común, como un JOIN de SQL. Con `pd.concat` apilás tablas, ya sea por filas (`axis=0`, el default) o por columnas (`axis=1`). Con `set_index` convertís una columna en el índice del DataFrame, y con `reset_index` volvés atrás ese cambio. Estas herramientas son clave cuando los datos vienen repartidos en varias tablas y hay que juntarlos antes de analizarlos.
