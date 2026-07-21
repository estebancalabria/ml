# LAB-0900 – Clasificación - ML: Titanic con Random Forest y KNN

## Objetivo

Armar un pipeline completo de clasificación sobre el dataset Titanic: explorar los datos, eliminar columnas que no aportan, tratar valores nulos, codificar variables categóricas, separar en train/test, y comparar dos modelos (Random Forest y KNN) con las métricas correctas.

## Requisitos previos

```
pip install scikit-learn matplotlib pandas
```

## El dataset

Vamos a usar `titanic.csv`, con 418 pasajeros y estas columnas:

| Columna | Significado |
|---|---|
| PassengerId | Identificador del pasajero |
| Survived | 1 = sobrevivió, 0 = no sobrevivió (nuestro **label**) |
| Pclass | Clase del ticket (1, 2 o 3) |
| Name | Nombre completo |
| Sex | Sexo |
| Age | Edad |
| SibSp | Hermanos/cónyuges a bordo |
| Parch | Padres/hijos a bordo |
| Ticket | Número de ticket |
| Fare | Precio pagado |
| Cabin | Cabina |
| Embarked | Puerto de embarque (C, Q o S) |

---

## Ejercicio 1 – Carga y exploración inicial

### Código

```python
import pandas as pd

df = pd.read_csv("titanic.csv")

print(df.shape)
df.info()
print(df.isnull().sum())
```

### Salida

```
(418, 12)
...
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age             86
SibSp            0
Parch            0
Ticket           0
Fare             1
Cabin          327
Embarked         0
dtype: int64
```

### Explicación línea por línea

- `df.shape`: nos dice que tenemos 418 filas (pasajeros) y 12 columnas.
- `df.info()`: muestra el tipo de dato de cada columna. Nos sirve para detectar qué es numérico y qué es texto.
- `df.isnull().sum()`: cuenta los nulos por columna. Acá vemos el primer problema: `Age` tiene 86 nulos, `Fare` tiene 1, y `Cabin` tiene 327 sobre 418 filas (¡el 78%!).

### Consigna

¿Qué columna te parece más problemática por su cantidad de nulos? ¿Tiene sentido intentar "rellenarla" o es mejor directamente descartarla?

---

## Ejercicio 2 – Eliminación de columnas que no aportan

Antes de entrenar cualquier modelo, hay que sacar las columnas que no sirven como features. Los motivos para descartar una columna suelen ser:

- **Es un identificador** (no tiene relación causal con el resultado): `PassengerId`.
- **Tiene texto libre con altísima cardinalidad** (casi un valor distinto por fila, imposible de codificar bien): `Name`, `Ticket`.
- **Tiene demasiados nulos como para confiar en una imputación**: `Cabin` (78% vacía).

### Código

```python
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

print(df.dtypes)
print(df.isnull().sum())
```

### Salida

```
Survived      int64
Pclass        int64
Sex          object
Age         float64
SibSp         int64
Parch         int64
Fare        float64
Embarked     object
dtype: object

Survived     0
Pclass       0
Sex          0
Age         86
Fare         1
Embarked     0
```

### Explicación línea por línea

- `df.drop(columns=[...])`: elimina esas 4 columnas del DataFrame. Nos quedamos con las que sí tienen potencial predictivo: `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`.
- Fijate que después de este drop, los únicos nulos que quedan son `Age` (86) y `Fare` (1). El problema de `Cabin` ya lo resolvimos sacándola.

### Consigna

¿Por qué `Pclass` no se descarta a pesar de ser un número (1, 2 o 3) que en el fondo representa una categoría? Pensalo en términos de si tiene un orden lógico.

---

## Ejercicio 3 – Manejo de valores nulos

`Age` y `Fare` son numéricas, así que las vamos a rellenar con la **mediana** (más robusta que el promedio ante valores extremos).

### Código

```python
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

print(df.isnull().sum())
```

### Salida

```
Survived    0
Pclass      0
Sex         0
Age         0
SibSp       0
Parch       0
Fare        0
Embarked    0
dtype: int64
```

### Explicación línea por línea

- `df["Age"].median()`: calcula la mediana de las edades conocidas (ignora los nulos automáticamente).
- `.fillna(...)`: reemplaza cada `NaN` de esa columna por el valor que le pasamos.
- Repetimos lo mismo con `Fare`, que tenía un solo nulo.
- Al final, `isnull().sum()` nos confirma que ya no queda ningún valor faltante.

> [!NOTE]
> Podríamos haber usado la media en vez de la mediana. La mediana es preferible cuando sospechamos outliers (por ejemplo, algún pasajero pagó un `Fare` carísimo comparado con el resto, y eso "tira" el promedio para arriba).

### Consigna

¿Qué hubiera pasado si en vez de imputar hubiésemos hecho `dropna()` sobre `Age`? ¿Cuántas filas hubiéramos perdido?

---

## Ejercicio 4 – Encoding de variables categóricas

Nos quedan dos columnas de texto: `Sex` (2 valores) y `Embarked` (3 valores). Ningún modelo de scikit-learn entrena con texto, así que hay que codificarlas.

### Código

```python
from sklearn.preprocessing import OneHotEncoder

print(df["Sex"].unique())
print(df["Embarked"].unique())

# Sex: solo 2 categorías, drop="first" para quedarnos con una sola columna binaria
codificador_sexo = OneHotEncoder(sparse_output=False, drop="first")
df["Sex_male"] = codificador_sexo.fit_transform(df[["Sex"]])

# Embarked: 3 categorías
codificador_embarked = OneHotEncoder(sparse_output=False, drop="first")
columnas_embarked = codificador_embarked.fit_transform(df[["Embarked"]])
nombres_embarked = codificador_embarked.get_feature_names_out(["Embarked"])
df[nombres_embarked] = columnas_embarked

df = df.drop(columns=["Sex", "Embarked"])

print(df.head())
```

### Salida

```
['male' 'female']
['Q' 'S' 'C']

   Survived  Pclass   Age  SibSp  Parch    Fare  Sex_male  Embarked_Q  Embarked_S
0         0       3  34.5      0      0  7.8292       1.0         1.0         0.0
1         1       3  47.0      1      0  7.0000       0.0         0.0         1.0
2         0       2  62.0      0      0  9.6875       1.0         1.0         0.0
3         0       3  27.0      0      0  8.6625       1.0         0.0         1.0
4         1       3  22.0      1      1 12.2875       0.0         0.0         1.0
```

### Explicación línea por línea

- `Sex` tiene solo 2 categorías → con `drop="first"` nos alcanza una sola columna (`Sex_male`: 1 = hombre, 0 = mujer).
- `Embarked` tiene 3 categorías → `OneHotEncoder` con `drop="first"` genera 2 columnas nuevas (`Embarked_Q`, `Embarked_S`); el puerto `C` queda representado implícitamente cuando ambas dan 0.
- Al final borramos las columnas de texto originales (`Sex`, `Embarked`) porque ya están representadas en las nuevas columnas numéricas.
- Ahora **todo** el DataFrame es numérico, condición necesaria para entrenar.

### Consigna

¿Por qué no usamos `OrdinalEncoder` para `Embarked`? Pensalo en términos de si los puertos C, Q y S tienen un orden natural entre sí.

---

## Ejercicio 5 – Definir X, y y hacer el Train/Test Split

### Código

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=["Survived"])
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("Train:", X_train.shape)
print("Test:", X_test.shape)
```

### Salida

```
Train: (292, 8)
Test: (126, 8)
```

### Explicación línea por línea

- `X`: todas las columnas menos `Survived` (nuestras features).
- `y`: la columna `Survived` (lo que queremos predecir).
- `train_test_split(..., test_size=0.3, random_state=42)`: separa 70% para entrenar y 30% para probar. `random_state=42` asegura que el split sea siempre el mismo, para que los resultados sean reproducibles.

### Consigna

¿Por qué es importante que el modelo nunca "vea" los datos de `X_test` durante el entrenamiento?

---

## Ejercicio 6 – Entrenar Random Forest

### Código

```python
from sklearn.ensemble import RandomForestClassifier

modelo_rf = RandomForestClassifier(random_state=42)
modelo_rf.fit(X_train, y_train)

pred_rf = modelo_rf.predict(X_test)
```

### Explicación línea por línea

- `RandomForestClassifier`: entrena muchos árboles de decisión distintos (cada uno con una muestra al azar de los datos y de las columnas) y después promedia/vota sus predicciones. Por eso suele generalizar mejor que un único árbol.
- `random_state=42`: fija la semilla aleatoria, para que el resultado sea reproducible.
- `.fit(X_train, y_train)`: entrena.
- `.predict(X_test)`: genera las predicciones sobre los datos que el modelo nunca vio.

### Consigna

Recordá de la Clase 3: los modelos basados en árboles "zafan" bien de la colinealidad entre features. ¿Te parece que acá hay riesgo de colinealidad entre alguna de nuestras columnas?

---

## Ejercicio 7 – Entrenar KNN

### Código

```python
from sklearn.neighbors import KNeighborsClassifier

modelo_knn = KNeighborsClassifier(n_neighbors=5)
modelo_knn.fit(X_train, y_train)

pred_knn = modelo_knn.predict(X_test)
```

### Explicación línea por línea

- `KNeighborsClassifier(n_neighbors=5)`: para cada pasajero de test, busca los 5 pasajeros de entrenamiento más "cercanos" (según la distancia entre sus features) y predice la clase mayoritaria entre ellos.
- Acá hay un detalle importante que no tuvimos con Random Forest: KNN mide **distancias**, así que es sensible a la escala de las variables (`Age` va de 0 a 80, `Fare` puede llegar a cientos, `Sex_male` es 0 o 1). En un laboratorio más avanzado, a KNN convendría escalarle las features con `StandardScaler` antes de entrenar. Por ahora lo dejamos así a propósito, para poder comparar el impacto.

### Consigna

¿Qué modelo te imaginás que va a rendir peor por no tener las variables escaladas: Random Forest o KNN? ¿Por qué?

---

## Ejercicio 8 – Comparar métricas

### Código

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

for nombre, predicciones in [("Random Forest", pred_rf), ("KNN", pred_knn)]:
    print(f"--- {nombre} ---")
    print("Accuracy :", accuracy_score(y_test, predicciones))
    print("Precision:", precision_score(y_test, predicciones))
    print("Recall   :", recall_score(y_test, predicciones))
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, predicciones))
    print()
```

### Salida

```
--- Random Forest ---
Accuracy : 1.0
Precision: 1.0
Recall   : 1.0
Matriz de confusión:
[[85  0]
 [ 0 41]]

--- KNN ---
Accuracy : 0.6905
Precision: 0.5333
Recall   : 0.3902
Matriz de confusión:
[[71 14]
 [25 16]]
```

### Consigna

A simple vista Random Forest "ganó" con un 100% de accuracy. Antes de festejar... ¿un modelo perfecto en un problema de predicción sobre personas te genera confianza, o sospecha?

---

## Ejercicio 9 (el más importante) – Investigar el 100% de accuracy

Un modelo con accuracy, precision y recall en 1.0 sobre un dataset real de personas es una señal de alarma, no un logro. Investiguemos por qué pasa.

### Código

```python
import pandas as pd

df_original = pd.read_csv("titanic.csv")

print(df_original.groupby("Sex")["Survived"].mean())
print(df_original.groupby(["Sex", "Pclass"])["Survived"].mean())
```

### Salida

```
Sex
female    1.0
male      0.0
Name: Survived, dtype: float64

Sex     Pclass
female  1         1.0
        2         1.0
        3         1.0
male    1         0.0
        2         0.0
        3         0.0
```

### Explicación

En este `titanic.csv` en particular, `Survived` es **100% determinado por `Sex`**: todas las mujeres tienen `Survived = 1` y todos los hombres `Survived = 0`, sin excepciones. No es que Random Forest haya "aprendido" un patrón sutil del comportamiento humano en un naufragio: encontró una regla perfecta y trivial (`Sex_male == 0 → sobrevive`) porque el propio dataset la tiene incrustada.

Esto suele pasar cuando el archivo `titanic.csv` que se descarga de internet es en realidad el `test.csv` de Kaggle combinado con el `gender_submission.csv` (el archivo de ejemplo que Kaggle da como baseline, que literalmente predice "sobrevive si es mujer"), en vez del dataset histórico real con los desenlaces verdaderos.

> [!NOTE]
> Esta es la razón exacta por la que siempre conviene desconfiar de un modelo con métricas perfectas: o hay **data leakage** (el label se filtró en alguna feature), o el dataset no es lo que pensábamos que era. KNN, con su 69% de accuracy, en este caso terminó siendo el "canario en la mina" que nos hizo notar algo raro (si ni siquiera un modelo simple da bajo con una regla tan directa como el sexo, algo no cierra).

### Consigna final

Si consiguieran el dataset real de entrenamiento del Titanic (`train.csv` de Kaggle, con los desenlaces históricos verdaderos y no el baseline), ¿qell accuracy esperarían de Random Forest? Volvé a correr todo el laboratorio con ese archivo y compará.

---

## Ejercicio 10 – Matriz de correlación (colinealidad)

Como vimos en la Clase 3, la matriz de correlación nos sirve para dos cosas: entender qué variables se relacionan con el `Survived` (lo que queremos predecir) y detectar colinealidad entre features (dos columnas que aportan casi la misma información).

### Código

```python
import matplotlib.pyplot as plt

matriz_correlacion = df.corr()

print(matriz_correlacion.round(2))

plt.figure(figsize=(7, 6))

plt.imshow(matriz_correlacion, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlación")

plt.xticks(range(len(matriz_correlacion.columns)), matriz_correlacion.columns, rotation=45, ha="right")
plt.yticks(range(len(matriz_correlacion.columns)), matriz_correlacion.columns)

for i in range(len(matriz_correlacion)):
    for j in range(len(matriz_correlacion)):
        plt.text(j, i, f"{matriz_correlacion.iloc[i, j]:.2f}", ha="center", va="center", color="black")

plt.title("Matriz de Correlación - Titanic")
plt.tight_layout()
plt.show()
```

> [!NOTE]
> Este código va después del Ejercicio 4 (una vez que `df` ya tiene todas las columnas numéricas por el encoding). Si lo corrés después del Ejercicio 5 en adelante, acordate de que `df` sigue intacto: el split creó `X_train`/`X_test` a partir de él, pero no lo modificó.

### Salida

```
            Survived  Pclass   Age  SibSp  Parch  Fare  Sex_male  Embarked_Q  Embarked_S
Survived        1.00   -0.11  0.01   0.10   0.16  0.19     -1.00        0.12       -0.11
Pclass         -0.11    1.00 -0.47   0.00   0.02 -0.58      0.11        0.25        0.12
Age             0.01   -0.47  1.00  -0.07  -0.04  0.34     -0.01       -0.04       -0.12
SibSp           0.10    0.00 -0.07   1.00   0.31  0.17     -0.10       -0.10        0.08
Parch           0.16    0.02 -0.04   0.31   1.00  0.23     -0.16       -0.13        0.09
Fare            0.19   -0.58  0.34   0.17   0.23  1.00     -0.19       -0.16       -0.18
Sex_male       -1.00    0.11 -0.01  -0.10  -0.16 -0.19      1.00       -0.12        0.11
Embarked_Q      0.12    0.25 -0.04  -0.10  -0.13 -0.16     -0.12        1.00       -0.47
Embarked_S     -0.11    0.12 -0.12   0.08   0.09 -0.18      0.11       -0.47        1.00
```

### Explicación línea por línea

- `df.corr()`: calcula el coeficiente de correlación de Pearson entre cada par de columnas numéricas. Va de -1 (relación lineal inversa perfecta) a 1 (relación lineal directa perfecta), pasando por 0 (sin relación lineal).
- `plt.imshow(..., cmap="coolwarm", vmin=-1, vmax=1)`: pinta la matriz como un mapa de calor, fijando la escala de color entre -1 y 1 para que los colores sean comparables entre distintos gráficos.
- El doble `for` con `plt.text(...)` escribe el valor numérico dentro de cada celda, para no depender solo del color.

### Lectura de la matriz

- **`Survived` vs `Sex_male` = -1.00**: correlación lineal perfecta. Esto confirma en números exactos lo que descubrimos "a mano" en el Ejercicio 9 — acá no hace falta ni entrenar un modelo para verlo, la matriz ya nos lo grita.
- **`Pclass` vs `Fare` = -0.58**: tiene sentido, a mejor clase (número más bajo) el pasajero pagó más caro. No es un problema grave de colinealidad, pero es esperable.
- **`Age` vs `Pclass` = -0.47**: los pasajeros de primera clase tendían a ser un poco mayores.
- El resto de las variables tiene correlaciones bajas entre sí (todas menores a 0.35 en valor absoluto), así que no hay indicios fuertes de colinealidad entre features que nos obligue a descartar alguna columna por ese motivo.

### Consigna

Si `Sex_male` tiene correlación -1.00 con `Survived`, y ya sabemos por el Ejercicio 9 que ese dataset tiene el label "roto" (viene del baseline de Kaggle, no del desenlace real), ¿qué esperarían que pase con esa correlación si corrieran este mismo ejercicio sobre el `train.csv` real, con los desenlaces históricos del naufragio? ¿Seguiría siendo -1.00, o bajaría a un valor más "realista" (fuerte, pero no perfecto)?

---

## Cierre del laboratorio

Recorrimos el pipeline completo: exploración de nulos → eliminación de columnas irrelevantes → imputación → encoding → train/test split → entrenamiento de dos modelos distintos → comparación de métricas → matriz de correlación. Y de yapa, un caso real de por qué nunca hay que confiar ciegamente en un accuracy de 100%: la matriz de correlación terminó siendo la confirmación numérica exacta de algo que ya habíamos sospechado a mano — siempre hay que preguntarse *por qué* el modelo es tan bueno antes de asumir que lo es.
