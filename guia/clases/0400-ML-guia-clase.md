# Guía de Clase 4 - Exploración de Datos, Encoding de Variables Categóricas y Árboles de Decisión

## Objetivo

Al finalizar la clase los alumnos deberían:

- Repasar los conceptos de Recall y Precision vistos en la clase anterior.
- Comprender la importancia de la exploración de datos antes de entrenar un modelo.
- Utilizar Pandas para inspeccionar y comprender un dataset.
- Entender por qué los algoritmos de Machine Learning requieren variables numéricas.
- Diferenciar los principales métodos de encoding de variables categóricas.
- Aplicar Label Encoding, Ordinal Encoding y One-Hot Encoding utilizando Scikit-Learn.
- Entrenar un árbol de decisión sobre un dataset con variables categóricas.
- Visualizar e interpretar el árbol generado por Scikit-Learn.

---

# 1. Repaso de la clase anterior

Repasar brevemente los conceptos principales:

- Matriz de confusión.
- Verdaderos positivos, falsos positivos, verdaderos negativos y falsos negativos.
- Recall.
- Precision.
- Cuándo conviene priorizar cada métrica según el problema de negocio.

No repetir toda la teoría, solamente recordar los conceptos necesarios para los ejercicios de la clase.

---

# 2. Laboratorio: Precision vs Recall (Lab 910)

Resolver el laboratorio utilizando el dataset provisto.

Objetivos del laboratorio:

- Entrenar un clasificador.
- Obtener la matriz de confusión.
- Calcular Recall y Precision.
- Comparar ambas métricas.
- Analizar distintos escenarios donde una métrica resulta más importante que la otra.

Utilizar las funciones de Scikit-Learn para calcular ambas métricas.

Al finalizar discutir:

- ¿Qué significa obtener un Recall alto?
- ¿Qué significa obtener un Precision alto?
- ¿Cuál elegiríamos para un detector de enfermedades?
- ¿Cuál elegiríamos para un detector de spam?

---

# 3. Kahoot

Realizar un Kahoot de repaso sobre:

- Matriz de confusión.
- Accuracy.
- Recall.
- Precision.
- Falsos positivos.
- Falsos negativos.

Resolver las dudas que aparezcan antes de continuar.

---

# 4. Exploración de datos con Pandas

Antes de construir cualquier modelo es fundamental conocer el dataset.

Trabajar con los laboratorios de Pandas para realizar una exploración completa.

Utilizar operaciones como:

- Carga del dataset.
- Visualización de registros.
- Información general del DataFrame.
- Estadísticas descriptivas.
- Tipos de datos.
- Valores nulos.
- Cantidad de categorías por columna.
- Frecuencia de valores.
- Filtrado de datos.
- Selección de columnas.
- Creación de nuevas columnas.

Ejemplos:

```python
df.head()

df.info()

df.describe()

df["Genero"].value_counts()

df.isnull().sum()
```

Discutir:

- ¿Qué columnas son numéricas?
- ¿Qué columnas son categóricas?
- ¿Cuáles podrían necesitar limpieza?
- ¿Qué variables podrían utilizarse para entrenar un modelo?

---

# 5. Encoding de variables categóricas con Scikit-Learn (Lab 0620)

## ¿Por qué necesitamos encoding?

Explicar que la mayoría de los algoritmos de Machine Learning solamente trabajan con números.

Los modelos no comprenden textos como:

- Rojo
- Azul
- Verde

o

- Soltero
- Casado
- Divorciado

Es necesario transformar estas categorías en representaciones numéricas.

---

## Label Encoding

Explicar que asigna un número entero distinto a cada categoría.

Ejemplo:

```
Rojo  -> 0
Azul  -> 1
Verde -> 2
```

Mostrar su utilización con Scikit-Learn.

Discutir cuándo conviene utilizarlo y cuándo puede inducir un orden inexistente.

---

## Ordinal Encoding

Explicar que se utiliza cuando las categorías poseen un orden natural.

Ejemplo:

```
Bajo
Medio
Alto
```

o

```
Primario
Secundario
Universitario
```

Mostrar su implementación utilizando Scikit-Learn.

---

## One-Hot Encoding

Explicar que crea una columna para cada categoría.

Ejemplo:

```
Color_Rojo

Color_Azul

Color_Verde
```

Cada fila tendrá un único valor igual a 1.

Explicar:

- Ventajas.
- Desventajas.
- Incremento de dimensionalidad.
- Cuándo es el método recomendado.

---

## Comparación de los tres métodos

Comparar:

- Información que conserva cada técnica.
- Casos de uso.
- Ventajas.
- Desventajas.
- Algoritmos donde suelen utilizarse.

---

# 6. Kahoot

Repasar mediante Kahoot:

- Variables categóricas.
- Label Encoding.
- Ordinal Encoding.
- One-Hot Encoding.
- Cuándo utilizar cada técnica.

---

# 7. Árbol de decisión con variables categóricas (Lab 720)

Realizar un ejemplo completo donde:

- Se carga un dataset.
- Se identifican las variables categóricas.
- Se aplica el encoding correspondiente.
- Se divide el dataset utilizando `train_test_split`.
- Se entrena un árbol de decisión.
- Se realizan predicciones.

Ejemplo:

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

predicciones = model.predict(X_test)
```

---

## Visualización del árbol

Mostrar cómo Scikit-Learn permite representar gráficamente el árbol entrenado.

Ejemplo:

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(18,10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=True,
    filled=True
)

plt.show()
```

Explicar cómo interpretar el gráfico:

- Nodo raíz.
- Condición de cada división.
- Gini.
- Samples.
- Value.
- Clase predicha.
- Hojas del árbol.

Relacionar el árbol obtenido con el funcionamiento explicado teóricamente en la clase anterior.

---

# 8. Cierre

Repasar los conceptos aprendidos:

- Exploración inicial de un dataset.
- Identificación de variables categóricas.
- Necesidad del encoding.
- Label Encoding.
- Ordinal Encoding.
- One-Hot Encoding.
- Entrenamiento de un árbol de decisión.
- Interpretación visual del árbol generado por Scikit-Learn.

Adelantar la próxima clase:

> Compararemos distintos algoritmos de clasificación basados en árboles, como Random Forest y Gradient Boosting, analizando cómo mejoran el rendimiento respecto a un único árbol de decisión.

Responder preguntas.

---

# Checklist antes de terminar

- Repaso de Recall y Precision realizado.
- Laboratorio de métricas completado.
- Kahoot de métricas realizado.
- Exploración del dataset con Pandas completada.
- Variables categóricas identificadas.
- Label Encoding aplicado.
- Ordinal Encoding aplicado.
- One-Hot Encoding aplicado.
- Diferencias entre los tres métodos comprendidas.
- Árbol de decisión entrenado.
- Árbol visualizado e interpretado.
- Preguntas respondidas.
