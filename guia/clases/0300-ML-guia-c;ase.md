# Guía de Clase 3 - Colinealidad, Métricas de Clasificación y Práctica con Pandas

## Objetivo

Al finalizar la clase los alumnos deberían: - Entender qué es la
correlación y su relación matemática con la regresión lineal simple. -
Interpretar una matriz de correlación y saber detectar
multicolinealidad. - Comprender qué problemas trae la multicolinealidad
y cuándo conviene reducir dimensionalidad. - Entender Recall y Precision
como métricas derivadas de la matriz de confusión, y cuándo priorizar
cada una. - Practicar manipulación de datos con Pandas. - Ejecutar un
ejemplo completo de clasificación y uno de regresión, con sus gráficos
correspondientes.

------------------------------------------------------------------------

# 1. Repaso (15-20 min)

Repasar rápido lo visto en clase 2:

-   Librerías del ecosistema ML (NumPy, Pandas, Matplotlib, Seaborn,
    Scikit-Learn).
-   Gráficos de visualización: histograma, boxplot, scatter, heatmap
    de correlación (visto de forma demostrativa, sin profundizar).
-   Diferencia entre error (regresión) y acierto/fallo (clasificación).
-   Matriz de confusión: TP, TN, FP, FN.

No repetir el código, solo los conceptos.

------------------------------------------------------------------------

# 2. Colinealidad: de la regresión lineal a la matriz de correlación (60-70 min)

## Regresión lineal simple y error (10 min)

Recordar rápido: ajustamos una recta (Y = β₀ + β₁·X) y medimos qué tan
bien ajusta con MSE/RMSE. Problema: el error depende de la escala de
los datos, no es comparable entre variables distintas.

## R² (10 min)

Normaliza el error en una proporción de varianza explicada, entre 0 y
1. Al ser adimensional, sí es comparable entre variables.

## La conexión R² = r² (10 min)

En regresión simple, el R² es literalmente el coeficiente de
correlación de Pearson al cuadrado. Calcular r entre dos variables
equivale a ajustar una recta entre ellas y ver qué tan bien explica.

## Matriz de correlación (15-20 min)

Retomar el heatmap que se mostró de forma demostrativa en clase 2, pero
ahora explicando qué representa: el resumen de calcular r para todos
los pares de variables del dataset a la vez.

```python
import seaborn as sns
import matplotlib.pyplot as plt

correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Matriz de correlación")
plt.show()
```

Se lee en dos sentidos:

-   Fila/columna del target: qué variables aportan señal predictiva.
-   Pares entre features: qué variables son redundantes entre sí.

## Multicolinealidad (15-20 min)

Qué pasa cuando dos predictoras están muy correlacionadas entre sí
(ejemplo: metros cuadrados y cantidad de ambientes).

Dificultades que trae:

-   **Explicabilidad**: no se puede confiar en el coeficiente
    individual de una variable.
-   **Inestabilidad de coeficientes**: el modelo reparte el "crédito"
    de forma arbitraria entre variables correlacionadas.
-   **Errores estándar inflados**: se pierde confianza sobre el peso
    real de cada variable.

No afecta a todos los algoritmos por igual: modelos lineales y basados
en distancia (KNN, SVM) sufren fuerte; árboles y ensembles (Random
Forest, XGBoost) casi no se ven afectados en la predicción, aunque sí
se les distorsiona la feature importance.

## Reducción de dimensionalidad (5 min)

De cada par muy correlacionado, quedarse con la variable de mayor
correlación con el target y descartar la otra.

------------------------------------------------------------------------

# --- BREAK (15 min) ---

------------------------------------------------------------------------

# 3. Errores en clasificación: Recall y Precision (40-50 min)

## Repaso de matriz de confusión (10 min)

Recordar TP, TN, FP, FN con el ejemplo de aprobación/rechazo o
enfermo/sano visto en clase 2.

## El problema del Accuracy (10 min)

Mensaje: contar aciertos totales no siempre alcanza. Ejemplo con clases
desbalanceadas (ej. 95% sanos, 5% enfermos): un modelo que siempre
predice "sano" tiene 95% de accuracy pero es inútil.

> **¿Cómo medimos algo más específico que "cuántos acerté en total"?**

## Recall (Sensibilidad) (10-15 min)

De todos los casos que **realmente eran positivos**, ¿cuántos detectó
el modelo?

```
Recall = TP / (TP + FN)
```

Importa cuando el costo de un falso negativo es alto (ej. no detectar
una enfermedad).

## Precision (10-15 min)

De todos los casos que el modelo **predijo como positivos**, ¿cuántos
lo eran realmente?

```
Precision = TP / (TP + FP)
```

Importa cuando el costo de un falso positivo es alto (ej. marcar un
mail legítimo como spam).

## El trade-off (5 min)

Mensaje: generalmente subir Recall baja Precision y viceversa. La
elección depende del problema de negocio, igual que la elección de
algoritmo en la clase anterior.

```python
from sklearn.metrics import recall_score, precision_score

recall_score(y_real, y_pred)
precision_score(y_real, y_pred)
```

------------------------------------------------------------------------

# 4. Ejercicios con Pandas (40-50 min)

Práctica guiada sobre un dataset del curso:

-   Carga del dataset (`pd.read_csv`).
-   Exploración inicial (`.head()`, `.info()`, `.describe()`).
-   Detección de nulos (`.isnull().sum()`).
-   Filtrado y selección de columnas.
-   Creación de columnas derivadas.
-   Cálculo de la matriz de correlación sobre el dataset propio
    (`df.corr()`).

Ejercicio en grupos: que cada grupo identifique en su dataset un par de
variables candidatas a multicolinealidad usando la matriz.

------------------------------------------------------------------------

# --- BREAK (10-15 min) ---

------------------------------------------------------------------------

# 5. Ejemplo de clasificación y regresión con gráficos (50-60 min)

## Ejemplo de regresión (20-25 min)

Entrenar una regresión lineal simple sobre el dataset del curso,
graficar la recta ajustada sobre el scatter de los datos reales.

```python
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

model = LinearRegression()
model.fit(X_train, y_train)

plt.scatter(X_test, y_test, label="Real")
plt.plot(X_test, model.predict(X_test), color="red", label="Predicción")
plt.legend()
plt.show()
```

## Ejemplo de clasificación (20-25 min)

Entrenar un árbol de decisión, generar la matriz de confusión y
calcular Recall y Precision.

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, recall_score, precision_score

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
```

## Cierre del bloque (5-10 min)

Comparar ambos ejemplos: en regresión medimos error (MSE/R²), en
clasificación medimos aciertos/fallos categorizados (Recall/Precision).

------------------------------------------------------------------------

# 6. Cierre (10 min)

Repasar:

-   Correlación, R² y su relación con la regresión lineal simple.
-   Matriz de correlación y multicolinealidad.
-   Recall y Precision como métricas derivadas de la matriz de
    confusión.
-   Práctica con Pandas.
-   Ejemplo completo de regresión y clasificación.

Adelantar la próxima clase:

> Trabajaremos con train_test_split, cross validation y compararemos
> distintos algoritmos de clasificación de forma más rigurosa.

Responder preguntas.

------------------------------------------------------------------------

# Checklist antes de terminar

-   Matriz de correlación interpretada con el grupo.
-   Multicolinealidad explicada con ejemplo concreto.
-   Recall y Precision calculados y comparados.
-   Ejercicios de Pandas completados por todos.
-   Ejemplo de regresión ejecutado y graficado.
-   Ejemplo de clasificación ejecutado con matriz de confusión.
-   Preguntas respondidas.
