# Guía de Clase 2 - Visualización de Datos y Primera Clasificación

## Objetivo

Al finalizar la clase los alumnos deberían: - Conocer las librerías
principales del ecosistema ML y dónde está su documentación. - Saber
en qué entornos se puede trabajar con ML (local, cloud, notebooks). -
Poder generar e interpretar los gráficos básicos de visualización de
datos. - Entender la diferencia entre error (regresión) y acierto/fallo
(clasificación). - Ejecutar un ejemplo simple de clasificación y leer
una matriz de confusión.

------------------------------------------------------------------------

# 1. Repaso (25-30 min)

Repasar rápido lo visto en clase 1:

-   Qué es Machine Learning.
-   Dataset, Features (X) y Labels (y).
-   Datos de entrenamiento vs. datos de prueba.
-   Regresión lineal como primer modelo.
-   Overfitting.
-   ML en la industria (flujo Business → Data Engineers → Data
    Scientists → ML Engineers).
-   Uso del Agente Analista para programar ML.

No repetir el código, solo los conceptos.

------------------------------------------------------------------------

# 2. Librerías de ML (20 min)

Explicar para qué sirve cada una, con link a documentación oficial.

-   NumPy -- https://numpy.org/doc/
-   Pandas -- https://pandas.pydata.org/docs/
-   Matplotlib -- https://matplotlib.org/stable/
-   Seaborn -- https://seaborn.pydata.org/
-   Scikit-Learn -- https://scikit-learn.org/stable/
-   PyTorch (mención) -- https://pytorch.org/docs/

Aclarar:

> La mayor parte del curso utilizará Pandas, Matplotlib/Seaborn y
> Scikit-Learn.

------------------------------------------------------------------------

# 3. Entornos para trabajar con ML (25-30 min)

Mostrar y comparar:

-   Local (VSCode + Jupyter) -- lo que venimos usando.
-   Google Colab -- https://colab.research.google.com/
-   Kaggle Notebooks -- https://www.kaggle.com/code
-   Azure Machine Learning -- https://azure.microsoft.com/products/machine-learning

Para cada uno: ventajas, cuándo conviene usarlo, GPU gratis (Colab/Kaggle)
vs. entorno productivo (Azure).

## Kahoot -- Uso de ML en el negocio (10-15 min)

Preguntas sobre casos de aplicación de ML en distintas industrias
(bancos, retail, salud, marketing, manufactura).

------------------------------------------------------------------------

# 4. Visualización de datos (60-70 min)

## Intro (10 min)

Por qué visualizar antes de modelar: entender la distribución de los
datos, detectar outliers, ver relaciones entre variables.

## Histograma (15-20 min)

-   Para qué sirve: ver la distribución de una variable.
-   Ejercicio con el grupo.

```python
import matplotlib.pyplot as plt

plt.hist(datos, bins=10)
plt.xlabel("Valor")
plt.ylabel("Frecuencia")
plt.title("Histograma")
plt.show()
```

## Boxplot (15-20 min)

-   Para qué sirve: ver mediana, cuartiles y outliers.
-   Ejercicio con el grupo.

```python
import seaborn as sns

sns.boxplot(y=datos)
plt.title("Boxplot")
plt.show()
```

## Scatter (10 min, demostrativo)

-   Para qué sirve: ver relación entre dos variables.

```python
plt.scatter(x, y)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Scatter")
plt.show()
```

## Heatmap de correlación (10 min, demostrativo)

-   Para qué sirve: ver qué variables están relacionadas entre sí.

```python
import seaborn as sns

sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Heatmap de correlación")
plt.show()
```

------------------------------------------------------------------------

# Kahoot -- Librerías (10-15 min)

Preguntas sobre NumPy, Pandas, Matplotlib, Seaborn, Scikit-Learn.

------------------------------------------------------------------------

# --- BREAK (15 min) ---

------------------------------------------------------------------------

# 5. Repaso de error en regresión (15-20 min)

Recordar:

-   Error absoluto: diferencia entre valor real y predicción.
-   En regresión el error es un número (¿cuánto me equivoqué?).

Mensaje:

> En clasificación no vamos a medir "cuánto me equivoqué" sino "acerté
> o no acerté la categoría".

------------------------------------------------------------------------

# 6. Ejemplo de clasificación simple (30-40 min)

Retomar el ejemplo de aprobación/rechazo (o spam/no spam) y entrenarlo
con un algoritmo de clasificación simple (árbol de decisión).

```python
from sklearn.tree import DecisionTreeClassifier

X = [...]  # features
y = [...]  # 0/1 o categorías

model = DecisionTreeClassifier()
model.fit(X, y)

prediccion = model.predict([[...]])
print(prediccion)
```

Aclarar: la salida ya no es un número continuo, es una clase.

------------------------------------------------------------------------

# 7. Matriz de confusión (25-30 min)

Explicar con el ejemplo de aprobación/rechazo:

-   Verdadero Positivo (TP)
-   Falso Positivo (FP)
-   Verdadero Negativo (TN)
-   Falso Negativo (FN)

```python
from sklearn.metrics import confusion_matrix

matriz = confusion_matrix(y_real, y_pred)
print(matriz)
```

Interpretar la matriz con el grupo, sin entrar en accuracy/precision/recall
todavía (eso queda para más adelante).

------------------------------------------------------------------------

# 8. Adelanto: Cross Validation / Split de datos (5 min)

Mensaje:

> Hasta ahora entrenamos y probamos con los mismos datos. La próxima
> clase vamos a separar datos de entrenamiento y de prueba de forma
> correcta, y vamos a ver cross validation.

(Si sobra tiempo, se puede introducir `train_test_split`; si no, queda
para clase 3.)

------------------------------------------------------------------------

# 9. Cierre (10 min)

Repasar:

-   Librerías del ecosistema ML.
-   Entornos disponibles para trabajar.
-   Gráficos de visualización (histograma, boxplot, scatter, heatmap).
-   Diferencia entre error (regresión) y clasificación.
-   Matriz de confusión.

Adelantar la próxima clase:

> Trabajaremos con split de datos, cross validation y compararemos
> distintos algoritmos de clasificación.

Responder preguntas.

------------------------------------------------------------------------

# Checklist antes de terminar

-   Todos generaron los 4 gráficos.
-   Kahoot de negocio realizado.
-   Kahoot de librerías realizado.
-   Ejemplo de clasificación ejecutado.
-   Matriz de confusión interpretada.
-   Preguntas respondidas.
