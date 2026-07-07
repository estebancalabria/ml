# LAB-0860 – ML: Clasificación con Gradient Boosting sobre Breast Cancer

## Objetivo

Entrenar modelos de boosting sobre el dataset Breast Cancer Wisconsin y comprender la idea central de los ensambles secuenciales:

> No se trata de que muchos árboles voten al mismo tiempo, sino de que cada nuevo árbol intenta corregir errores del modelo anterior.

En este laboratorio se introducen:

* AdaBoost, brevemente.
* Gradient Boosting.
* Comparación contra un Árbol de Decisión.
* Matriz de confusión.
* Accuracy, Precision, Recall y F1 Score.
* Importancia de variables.
* Relación entre `learning_rate` y `n_estimators`.
* XGBoost y LightGBM de forma conceptual.

***

## Requisitos previos

```bash
pip install scikit-learn matplotlib pandas
```

> Nota: no instalamos XGBoost ni LightGBM en este laboratorio. Los mencionamos conceptualmente para no complicar el entorno.

***

# El dataset

Usaremos el dataset Breast Cancer Wisconsin incluido en scikit-learn.

```python
from sklearn.datasets import load_breast_cancer

datos = load_breast_cancer()

X = datos.data
y = datos.target

print(X.shape)
print(y.shape)
print(datos.target_names)
print(datos.feature_names[:10])
```

### Salida esperada

```text
(569, 30)
(569,)
['malignant' 'benign']
['mean radius' 'mean texture' 'mean perimeter' ...]
```

## Explicación

* `X` contiene las variables numéricas del tumor.
* `y` contiene la clase.
* Las clases son:
  * `0 = malignant`
  * `1 = benign`
* Hay 30 features numéricas relacionadas con características del núcleo celular.

## Consigna

¿Por qué este dataset es más interesante que Iris para probar modelos más potentes como Gradient Boosting?

***

# Ejercicio 1 – Explorar el dataset

## Código

```python
import pandas as pd

df = pd.DataFrame(
    X,
    columns=datos.feature_names
)

df["target"] = y

print(df.head())
print(df["target"].value_counts())
print(datos.target_names)
```

## Explicación línea por línea

* Creamos un `DataFrame` para ver mejor los datos.
* Agregamos la columna `target`.
* Contamos cuántos casos hay de cada clase.

## Consigna

¿El dataset parece estar muy desbalanceado o razonablemente distribuido?

***

# Ejercicio 2 – Separar entrenamiento y prueba

## Código

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print(X_train.shape)
print(X_test.shape)
```

## Explicación línea por línea

* `test_size=0.3`: dejamos 30% para prueba.
* `random_state=42`: hace reproducible el resultado.
* `stratify=y`: mantiene proporciones similares de clases en train y test.

## Consigna

¿Por qué conviene usar `stratify=y` en problemas de clasificación?

***

# Ejercicio 3 – Modelo base: Árbol de Decisión

Antes de hacer boosting, entrenamos un árbol simple.

## Código

```python
from sklearn.tree import DecisionTreeClassifier

arbol = DecisionTreeClassifier(
    random_state=42
)

arbol.fit(
    X_train,
    y_train
)

print(
    "Accuracy árbol:",
    arbol.score(X_test, y_test)
)
```

## Explicación

Este modelo usa un único árbol para tomar decisiones.

Sirve como punto de comparación.

## Consigna

Anotá el accuracy del árbol. Después lo vamos a comparar con AdaBoost y Gradient Boosting.

***

# Ejercicio 4 – AdaBoost

AdaBoost es uno de los primeros algoritmos populares de boosting.

La idea es:

> Entrenar modelos débiles de forma secuencial, dando más importancia a los ejemplos que fueron mal clasificados.

## Código

```python
from sklearn.ensemble import AdaBoostClassifier

adaboost = AdaBoostClassifier(
    n_estimators=100,
    random_state=42
)

adaboost.fit(
    X_train,
    y_train
)

print(
    "Accuracy AdaBoost:",
    adaboost.score(X_test, y_test)
)
```

## Explicación línea por línea

* `AdaBoostClassifier()` crea un modelo de boosting.
* `n_estimators=100` indica la cantidad de modelos débiles.
* Cada nuevo modelo presta más atención a los errores anteriores.

## Consigna

¿AdaBoost mejora respecto al árbol simple?

***

# Ejercicio 5 – Gradient Boosting

Ahora entrenamos Gradient Boosting.

## Código

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

gb.fit(
    X_train,
    y_train
)

print(
    "Accuracy Gradient Boosting:",
    gb.score(X_test, y_test)
)
```

## Explicación línea por línea

* `GradientBoostingClassifier()` crea el modelo.
* `n_estimators=100` indica cuántos árboles secuenciales se entrenan.
* `learning_rate=0.1` controla cuánto aporta cada árbol nuevo.
* `max_depth=3` limita la profundidad de cada árbol.
* A diferencia de Random Forest, los árboles no son independientes: cada uno intenta mejorar el resultado acumulado.

## Concepto clave

Random Forest funciona así:

```text
Árbol 1 vota
Árbol 2 vota
Árbol 3 vota
Resultado final = voto mayoritario
```

Gradient Boosting funciona así:

```text
Modelo inicial se equivoca
Árbol 1 corrige parte del error
Árbol 2 corrige errores restantes
Árbol 3 vuelve a corregir
Resultado final = suma ordenada de mejoras
```

## Consigna

¿Gradient Boosting mejora respecto al árbol simple y AdaBoost?

***

# Ejercicio 6 – Comparar modelos con métricas

## Código

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

modelos = {
    "Árbol": arbol,
    "AdaBoost": adaboost,
    "Gradient Boosting": gb
}

resultados = []

for nombre, modelo in modelos.items():

    y_pred = modelo.predict(X_test)

    resultados.append(
        {
            "Modelo": nombre,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred)
        }
    )

tabla_resultados = pd.DataFrame(resultados)

print(tabla_resultados)
```

## Explicación línea por línea

* Entrenamos tres modelos.
* Para cada modelo calculamos:
  * Accuracy
  * Precision
  * Recall
  * F1 Score
* Guardamos todo en una tabla.

## Consigna

¿Cuál modelo parece mejor?  
¿Todos ganan en todas las métricas o alguno mejora una métrica pero empeora otra?

***

# Ejercicio 7 – Matriz de confusión

## Código

```python
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay.from_estimator(
    gb,
    X_test,
    y_test,
    display_labels=datos.target_names
)

plt.title("Matriz de confusión - Gradient Boosting")
plt.show()
```

## Explicación

La matriz de confusión muestra:

* Casos correctamente clasificados.
* Casos donde el modelo confundió malignant con benign.
* Casos donde confundió benign con malignant.

## Consigna

En un problema médico, ¿qué error sería más grave?

* Predecir benign cuando era malignant.
* Predecir malignant cuando era benign.

***

# Ejercicio 8 – Classification Report

## Código

```python
from sklearn.metrics import classification_report

y_pred_gb = gb.predict(X_test)

print(
    classification_report(
        y_test,
        y_pred_gb,
        target_names=datos.target_names
    )
)
```

## Explicación

`classification_report` muestra automáticamente:

* Precision
* Recall
* F1 Score
* Support

para cada clase.

## Consigna

¿El modelo tiene el mismo rendimiento para malignant y benign?

***

# Ejercicio 9 – Importancia de variables

Una ventaja de los modelos basados en árboles es que permiten obtener importancia de variables.

## Código

```python
importancias = pd.DataFrame(
    {
        "Feature": datos.feature_names,
        "Importancia": gb.feature_importances_
    }
)

importancias = importancias.sort_values(
    by="Importancia",
    ascending=False
)

print(importancias.head(10))
```

## Gráfico

```python
top_10 = importancias.head(10)

plt.figure(figsize=(10,6))

plt.barh(
    top_10["Feature"],
    top_10["Importancia"]
)

plt.gca().invert_yaxis()

plt.xlabel("Importancia")
plt.title("Top 10 variables más importantes - Gradient Boosting")

plt.show()
```

## Explicación

El modelo asigna mayor importancia a las variables que más ayudaron a separar las clases.

## Consigna

¿Qué variables aparecen como más importantes?  
¿Te parece razonable que esas variables ayuden a distinguir entre benign y malignant?

***

# Ejercicio 10 – Comparar cantidad de árboles

Ahora veremos qué pasa al cambiar `n_estimators`.

## Código

```python
valores_estimadores = [
    10,
    25,
    50,
    100,
    200,
    300
]

scores = []

for n in valores_estimadores:

    modelo_tmp = GradientBoostingClassifier(
        n_estimators=n,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

    modelo_tmp.fit(
        X_train,
        y_train
    )

    scores.append(
        modelo_tmp.score(
            X_test,
            y_test
        )
    )

plt.plot(
    valores_estimadores,
    scores,
    marker="o"
)

plt.xlabel("Cantidad de árboles")
plt.ylabel("Accuracy")
plt.title("Accuracy según cantidad de árboles")

plt.show()
```

## Explicación

* Pocos árboles pueden quedarse cortos.
* Muchos árboles pueden mejorar, pero también aumentar complejidad.
* En boosting, agregar árboles no siempre mejora indefinidamente.

## Consigna

¿A partir de cuántos árboles el accuracy deja de mejorar claramente?

***

# Ejercicio 11 – Comparar learning rate

`learning_rate` controla cuánto aporta cada árbol.

## Código

```python
valores_lr = [
    0.01,
    0.05,
    0.1,
    0.2,
    0.5,
    1.0
]

scores_lr = []

for lr in valores_lr:

    modelo_tmp = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=lr,
        max_depth=3,
        random_state=42
    )

    modelo_tmp.fit(
        X_train,
        y_train
    )

    scores_lr.append(
        modelo_tmp.score(
            X_test,
            y_test
        )
    )

plt.plot(
    valores_lr,
    scores_lr,
    marker="o"
)

plt.xlabel("Learning rate")
plt.ylabel("Accuracy")
plt.title("Accuracy según learning rate")

plt.show()
```

## Explicación

* `learning_rate` bajo: cada árbol aporta poco.
* `learning_rate` alto: cada árbol aporta mucho.
* Un learning rate demasiado alto puede hacer que el modelo aprenda de forma brusca.
* Un learning rate bajo suele necesitar más árboles.

## Consigna

¿Qué valor de `learning_rate` parece funcionar mejor en este caso?

***

# Ejercicio 12 – Relación entre learning rate y cantidad de árboles

Este es uno de los conceptos más importantes de Gradient Boosting.

## Código

```python
configuraciones = [
    {"n_estimators": 50, "learning_rate": 0.2},
    {"n_estimators": 100, "learning_rate": 0.1},
    {"n_estimators": 200, "learning_rate": 0.05},
    {"n_estimators": 500, "learning_rate": 0.01}
]

resultados_lr_estimadores = []

for config in configuraciones:

    modelo_tmp = GradientBoostingClassifier(
        n_estimators=config["n_estimators"],
        learning_rate=config["learning_rate"],
        max_depth=3,
        random_state=42
    )

    modelo_tmp.fit(
        X_train,
        y_train
    )

    score = modelo_tmp.score(
        X_test,
        y_test
    )

    resultados_lr_estimadores.append(
        {
            "n_estimators": config["n_estimators"],
            "learning_rate": config["learning_rate"],
            "Accuracy": score
        }
    )

tabla_lr = pd.DataFrame(resultados_lr_estimadores)

print(tabla_lr)
```

## Explicación

Existe una relación muy importante:

```text
Menor learning rate suele requerir más árboles.
Mayor learning rate suele requerir menos árboles.
```

Pero no hay una regla universal. Se prueba experimentalmente.

## Consigna

¿Qué combinación parece dar mejor resultado?

***

# Ejercicio 13 – Comparación visual final

## Código

```python
tabla_resultados.plot(
    x="Modelo",
    y=["Accuracy", "Precision", "Recall", "F1"],
    kind="bar",
    figsize=(10,6)
)

plt.ylim(0.8, 1.0)
plt.ylabel("Valor de la métrica")
plt.title("Comparación de modelos")
plt.xticks(rotation=0)

plt.show()
```

## Explicación

Este gráfico permite comparar rápidamente el rendimiento de los modelos.

## Consigna

Si tuvieras que elegir uno, ¿elegirías sólo por accuracy o mirarías también recall y precision?

***

# Ejercicio 14 – XGBoost y LightGBM conceptualmente

No los instalamos en este laboratorio, pero se presentan porque son muy usados en ML tabular.

## XGBoost

XGBoost puede pensarse como una versión optimizada y muy potente de Gradient Boosting.

Suele destacarse por:

* Buen rendimiento en datos tabulares.
* Regularización.
* Manejo eficiente de árboles.
* Buen desempeño en competiciones de Machine Learning.

## LightGBM

LightGBM también es una implementación moderna de boosting.

Suele destacarse por:

* Velocidad.
* Eficiencia con datasets grandes.
* Buen rendimiento en problemas tabulares.

## Comparación conceptual

| Modelo            | Idea principal                                       |
| ----------------- | ---------------------------------------------------- |
| AdaBoost          | Repondera errores anteriores                         |
| Gradient Boosting | Corrige errores usando gradientes                    |
| XGBoost           | Gradient Boosting optimizado y regularizado          |
| LightGBM          | Gradient Boosting optimizado para velocidad y escala |

## Consigna

¿Por qué creés que estos modelos se hicieron tan populares en datos tabulares empresariales?

***

# Cierre del laboratorio

En este laboratorio aprendiste:

* Qué es boosting.
* En qué se diferencia de Random Forest.
* Qué hace AdaBoost.
* Qué hace Gradient Boosting.
* Cómo comparar modelos con métricas.
* Cómo interpretar matriz de confusión.
* Cómo analizar importancia de variables.
* Cómo influyen `n_estimators` y `learning_rate`.
* Por qué XGBoost y LightGBM son tan usados en datos tabulares.

## Idea clave

Random Forest reduce el riesgo de error combinando muchos árboles independientes.

Gradient Boosting construye un modelo fuerte a partir de modelos débiles entrenados secuencialmente.

La diferencia pedagógica queda así:

```text
Árbol de Decisión:
Un árbol decide.

Random Forest:
Muchos árboles votan.

Gradient Boosting:
Muchos árboles se corrigen entre sí, uno después del otro.
```

## Frase final para la clase

> Gradient Boosting no intenta que todos los árboles sean excelentes de forma individual. Intenta que cada árbol aporte una pequeña mejora al error acumulado del modelo. Por eso puede ser tan potente en problemas tabulares.
