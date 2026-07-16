# Lab 0620 — Encoding de Variables Categóricas

## Objetivo

Convertir variables categóricas en variables numéricas que un modelo de ML pueda procesar, distinguiendo entre:
- Variables **ordinales** (tienen un orden natural)
- Variables **nominales** (no tienen orden)

Y entender por qué en un pipeline real (train/test) conviene usar `scikit-learn` en vez de `pandas.get_dummies()`.

## Dataset

Vamos a crear el dataset a mano, sin descargar nada externo.

```python
import pandas as pd

df = pd.DataFrame({
    'nivel_educativo': ['Secundario', 'Universitario', 'Posgrado', 'Universitario', 'Secundario'],
    'genero': ['M', 'F', 'F', 'M', 'F'],
    'default_pago': [0, 1, 0, 1, 0]
})

df
```

## Parte 1 — Encoding rápido con `get_dummies()`

Útil para exploración de datos (EDA), pero no para producción.

```python
df_dummies = pd.get_dummies(df, columns=['genero'], drop_first=True)
df_dummies
```

**Pregunta para la clase:** ¿Qué pasa si en el dataset de test aparece una categoría de `genero` que no vimos en train? (por ejemplo un valor `X` o `NB`). Probarlo y ver el resultado.

```python
df_test_ejemplo = pd.DataFrame({'genero': ['M', 'F', 'X']})
pd.get_dummies(df_test_ejemplo)
```

Noten que la columna resultante no coincide con la de train. Ese es el problema central de usar `get_dummies()` en un pipeline con train/test separados.

## Parte 2 — Encoding con `scikit-learn`

### 2.1 Ordinal Encoding

`nivel_educativo` tiene un orden natural: Secundario < Universitario < Posgrado.

```python
from sklearn.preprocessing import OrdinalEncoder

orden_educacion = [['Secundario', 'Universitario', 'Posgrado']]
ordinal_enc = OrdinalEncoder(categories=orden_educacion)

df['nivel_educativo_enc'] = ordinal_enc.fit_transform(df[['nivel_educativo']])
df
```

### 2.2 One-Hot Encoding

`genero` no tiene orden, así que usamos One-Hot.

```python
from sklearn.preprocessing import OneHotEncoder

onehot_enc = OneHotEncoder(sparse_output=False, drop='first')
genero_encoded = onehot_enc.fit_transform(df[['genero']])

df['genero_F'] = genero_encoded
df
```

## Parte 3 — El problema real: categorías nuevas en test

```python
from sklearn.model_selection import train_test_split

X = df[['genero']]
y = df['default_pago']

X_train, X_test = train_test_split(X, test_size=0.4, random_state=42)

# Simulamos una categoría nueva en test que no estaba en train
X_test_simulado = pd.DataFrame({'genero': ['F', 'X']})

onehot_enc = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
onehot_enc.fit(X_train)

# Sin handle_unknown='ignore' esto rompería con un ValueError
resultado = onehot_enc.transform(X_test_simulado)
resultado
```

**Punto clave para remarcar en clase:** `handle_unknown='ignore'` evita que el pipeline se rompa cuando aparece una categoría no vista, algo muy realista en un banco (clientes nuevos, categorías de producto nuevas, etc.).

## Cierre — Comparación

| Método | Fit/Transform separado | Maneja categorías nuevas | Uso recomendado |
|---|---|---|---|
| `pd.get_dummies()` | No | No | EDA / exploración rápida |
| `OneHotEncoder` (sklearn) | Sí | Sí (`handle_unknown='ignore'`) | Pipelines de producción |
| `OrdinalEncoder` (sklearn) | Sí | Sí (`handle_unknown='use_encoded_value'`) | Variables con orden natural |

## Ejercicio para los alumnos

1. Agregar una tercera variable categórica nominal al dataset (por ejemplo `tipo_credito`: Personal / Hipotecario / Prendario) y aplicar One-Hot Encoding.
2. Simular un split train/test de al menos 20 filas (pueden generar datos random con `numpy`) y verificar qué pasa si una categoría queda solo en test.
3. Discutir: ¿en qué caso usarían `get_dummies()` de todos modos?
