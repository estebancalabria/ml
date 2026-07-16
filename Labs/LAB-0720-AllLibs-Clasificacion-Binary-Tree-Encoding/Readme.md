# Lab 0720 — Árbol de Decisión con Encoding de Variables

## Objetivo

Armar un dataset chico a mano, aplicar **Label Encoding** y **One-Hot Encoding** según corresponda, entrenar un árbol de decisión (`DecisionTreeClassifier`) y visualizar el árbol resultante.

## Dataset

Vamos a simular datos de aprobación de crédito.

```python
import pandas as pd

df = pd.DataFrame({
    'nivel_educativo': ['Secundario', 'Universitario', 'Posgrado', 'Universitario',
                         'Secundario', 'Posgrado', 'Universitario', 'Secundario'],
    'tipo_empleo': ['Relacion_dependencia', 'Independiente', 'Relacion_dependencia', 'Independiente',
                     'Independiente', 'Relacion_dependencia', 'Relacion_dependencia', 'Independiente'],
    'ingreso_mensual': [45000, 62000, 90000, 55000, 38000, 105000, 71000, 41000],
    'aprobado': ['No', 'Si', 'Si', 'Si', 'No', 'Si', 'Si', 'No']
})

df
```

## Parte 1 — Ordinal Encoding

`nivel_educativo` tiene un orden natural: Secundario < Universitario < Posgrado. Le damos ese orden explícito con `OrdinalEncoder`, en vez de dejar que se asigne alfabéticamente.

```python
from sklearn.preprocessing import OrdinalEncoder

orden_educacion = [['Secundario', 'Universitario', 'Posgrado']]
ordinal_enc = OrdinalEncoder(categories=orden_educacion)

df['nivel_educativo_enc'] = ordinal_enc.fit_transform(df[['nivel_educativo']])

# La variable objetivo también la codificamos, pero como es binaria el orden no importa
orden_aprobado = [['No', 'Si']]
ordinal_target = OrdinalEncoder(categories=orden_aprobado)
df['aprobado_enc'] = ordinal_target.fit_transform(df[['aprobado']])

df
```

## Parte 2 — One-Hot Encoding

`tipo_empleo` no tiene orden, así que usamos `OneHotEncoder` en vez de Ordinal.

```python
from sklearn.preprocessing import OneHotEncoder

onehot_enc = OneHotEncoder(sparse_output=False, drop='first')
tipo_empleo_encoded = onehot_enc.fit_transform(df[['tipo_empleo']])

# El nombre de columna generado indica la categoría que quedó activa
nombre_columna = onehot_enc.get_feature_names_out(['tipo_empleo'])[0]
df[nombre_columna] = tipo_empleo_encoded

df
```

## Parte 3 — Armar el dataset final para el modelo

```python
X = df[['nivel_educativo_enc', 'ingreso_mensual', nombre_columna]]
y = df['aprobado_enc']

X
```

## Parte 4 — Entrenar el árbol de decisión

```python
from sklearn.tree import DecisionTreeClassifier

modelo = DecisionTreeClassifier(max_depth=3, random_state=42)
modelo.fit(X, y)
```

## Parte 5 — Visualizar el árbol

### Opción A: texto plano (rápido, sin dependencias gráficas)

```python
from sklearn.tree import export_text

reglas = export_text(modelo, feature_names=list(X.columns))
print(reglas)
```

### Opción B: gráfico con matplotlib

```python
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(12, 6))
plot_tree(modelo,
          feature_names=X.columns,
          class_names=['No aprobado', 'Aprobado'],
          filled=True,
          rounded=True)
plt.show()
```

## Cierre — Puntos para discutir en clase

1. Comparar la salida de `export_text` con el gráfico: ¿en qué columna hace el primer split el árbol? ¿Tiene sentido de negocio?
2. ¿Qué pasaría si hubiéramos usado `OrdinalEncoder` en `tipo_empleo` en vez de `OneHotEncoder`? (probarlo y ver si el árbol cambia sus splits de forma rara, por el orden numérico arbitrario que le impondría al no tener orden real)
3. Con `max_depth=3`, ¿el árbol está sobreajustando dado que el dataset tiene solo 8 filas?

## Ejercicio para los alumnos

1. Agregar 4 filas más al dataset con distintos valores de `ingreso_mensual` y `nivel_educativo`.
2. Repetir el entrenamiento con `max_depth=2` y comparar el árbol resultante.
3. Explicar con sus palabras por qué a `nivel_educativo` le conviene `OrdinalEncoder` (tiene orden real) mientras que a `tipo_empleo` le conviene `OneHotEncoder` (no tiene orden).
4. Probar `handle_unknown='ignore'` en el `OneHotEncoder` simulando una categoría nueva de `tipo_empleo` que no estaba en el dataset original (por ejemplo `'Monotributista'`).
