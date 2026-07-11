# Guía - Correlación y Multicolinealidad en la Preparación del Dataset

## Objetivos

Al finalizar esta guía podrás:

* Entender la diferencia entre problemas de regresión y clasificación, y por qué no afecta lo que vamos a ver.
* Comprender qué mide el coeficiente de correlación y cómo se relaciona con una regresión lineal simple.
* Interpretar una matriz de correlación como resumen de múltiples regresiones simples.
* Entender qué es la multicolinealidad y qué problemas concretos genera.
* Saber cuándo conviene reducir dimensionalidad eliminando variables correlacionadas, y cuándo no es necesario.

---

# 1. Tipos de problemas

En Machine Learning supervisado, distinguimos dos grandes tipos de problemas:

* **Regresión**: predecir un valor numérico continuo (ej. precio de una casa, temperatura, ingresos).
* **Clasificación**: predecir una categoría (ej. enfermo/sano, spam/no spam).

Todo lo que vamos a ver en esta guía —correlación y multicolinealidad— **aplica a ambos tipos de problema por igual**. No es una cuestión de si estamos haciendo regresión o clasificación, sino de qué **algoritmo** vamos a usar para resolver el problema.

---

# 2. Tipos de algoritmo

Antes de avanzar, conviene tener un panorama de las familias de algoritmos más comunes, porque más adelante en esta guía vamos a ver que la multicolinealidad no afecta a todas por igual.

* **Lineales**: Regresión Lineal, Regresión Logística
* **Basados en árboles**: Decision Tree, Random Forest, XGBoost / Gradient Boosting
* **Basados en distancia**: KNN, SVM (con kernel)
* **Probabilísticos**: Naive Bayes

---

# 3. Regresión lineal simple

Empecemos por el modelo más básico: ajustar una recta a los datos.

```
Y = β₀ + β₁·X
```

Donde β₁ es la pendiente de la recta y β₀ la ordenada al origen. El algoritmo busca los valores de β₀ y β₁ que mejor ajustan a los datos observados.

> **¿Cómo sabemos si esa recta ajusta bien?**

---

# 4. Error: MSE y RMSE

Para saber qué tan bien ajusta la recta, medimos cuánto se equivoca:

* **MSE** (Mean Squared Error): promedio de los errores al cuadrado.
* **RMSE** (Root MSE): la raíz del MSE, en las mismas unidades que la variable Y.

El problema de esta métrica es que **depende de la escala** de los datos. Si Y está en miles, el error absoluto es gigante; si está en decimales, es chiquito. Esto hace que no sea comparable entre variables distintas.

> **¿Cómo hacemos que el error sea comparable entre variables?**

---

# 5. R²: normalizando el error

El **R²** (coeficiente de determinación) normaliza el error en una proporción de varianza explicada, entre 0 y 1:

* R² = 1 → el modelo explica toda la variabilidad de los datos.
* R² = 0 → el modelo no explica nada.

A diferencia del MSE, el R² es adimensional, así que sí es comparable entre variables distintas.

---

# 6. La conexión: R² = r²

En una regresión lineal **simple** (una sola variable predictora), existe una igualdad muy útil:

```
R² = r²
```

Donde r es el coeficiente de correlación de Pearson. Es decir: el R² de ajustar una recta entre dos variables es literalmente el coeficiente de correlación al cuadrado.

Esto nos da una forma muy concreta de pensar la correlación: **calcular r entre dos variables es equivalente a calcular una regresión lineal simple entre ellas y ver qué tan bien ajusta**.

> **¿Y si quisiera ver esto para todos los pares de variables de mi dataset a la vez?**

---

# 7. La matriz de correlación

En vez de calcular r par por par a mano, podemos calcular todas las correlaciones de una:

```python
correlation_matrix = df.corr()
```

Y visualizarla como un mapa de calor:

```python
import seaborn as sns

sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
```

La matriz se lee en **dos sentidos**:

* **Fila/columna del target**: qué variables tienen mayor relación lineal con lo que queremos predecir → candidatas a aportar señal.
* **Pares entre features**: qué variables predictoras están muy correlacionadas entre sí → candidatas a ser redundantes.

---

# 8. Multicolinealidad

La **multicolinealidad** ocurre cuando dos (o más) variables predictoras están muy correlacionadas entre sí.

Ejemplo típico: metros cuadrados y cantidad de ambientes de una casa suelen moverse casi juntos. El modelo tiene dificultad para "separar" el efecto de una del efecto de la otra.

### Dificultades que trae

* **Explicabilidad**: no podés confiar en el coeficiente individual de una variable, porque su valor está contaminado por el de la otra variable correlacionada.
* **Inestabilidad de coeficientes**: el modelo no puede distinguir cuánto le corresponde a cada variable correlacionada, entonces reparte el "crédito" de forma arbitraria, y ese reparto cambia con cualquier variación mínima en los datos.
* **Errores estándar inflados**: al no poder separar el efecto de una variable del de la otra, el modelo pierde confianza sobre el peso real de cada una, aunque siga prediciendo bien en conjunto.
* **Feature importance distorsionada** (en árboles): la importancia se reparte entre las variables correlacionadas, y cada una aparece con menos peso del que tendría si estuviera sola.

### No afecta a todos los algoritmos por igual

| Afectado fuerte | Casi no afectado (en predicción) |
| --- | --- |
| Regresión lineal | Árboles / Random Forest / XGBoost |
| Regresión logística | |
| KNN / SVM (distancia) | |

Los modelos basados en árboles eligen una variable a la vez en cada split, así que si dos están correlacionadas, es indistinto cuál usen — el desempeño predictivo casi no se degrada. Lo que sí se distorsiona en árboles es la feature importance, no la calidad del modelo.

### El impacto depende del objetivo

* **Predicción pura con un modelo robusto** (árboles): la multicolinealidad casi no afecta.
* **Modelo lineal**: puede darte coeficientes inestables y poco interpretables.
* **Interpretación del modelo** (entender qué variable pesa cuánto): la multicolinealidad siempre puede confundir la lectura, sea cual sea el algoritmo.

> **Importante**: no detectar esto no significa "entrenar un modelo al pedo". El modelo entrena y predice igual. Lo que se pierde es la **confianza** que podemos tener en los coeficientes individuales o en la interpretación de importancia de variables.

---

# 9. Reducción de dimensionalidad

Con la matriz de correlación identificada, el criterio para decidir qué variable sacar es:

1. Buscar pares de features con correlación muy alta entre sí (ej. > 0.85–0.9).
2. De ese par, quedarse con la que tenga **mayor correlación con el target**.
3. Descartar la otra.

**Motivo principal**: evitar los problemas de multicolinealidad en los algoritmos que la sufren.

**Beneficio secundario**: menos features implica menos tiempo de entrenamiento y menor riesgo de overfitting.

---

# 10. Límite del método

La correlación (y por lo tanto la matriz de correlación) **solo captura relaciones lineales**.

Puede existir una relación fuerte y clara entre dos variables —por ejemplo cuadrática o cíclica— y aun así el coeficiente r dar bajo, porque esa relación no es una línea recta.

Por eso la matriz de correlación no reemplaza la inspección visual: conviene complementarla siempre con gráficos de dispersión (scatter plots) para no perderse relaciones no lineales importantes.

---

# 11. Resumen del recorrido

Cada paso de esta guía resuelve una limitación del anterior:

```
Regresión lineal
      ↓
Error (MSE/RMSE) → depende de la escala
      ↓
R² → normaliza el error, comparable entre variables
      ↓
R² = r² en regresión simple → conecta con correlación
      ↓
Matriz de correlación → generaliza r a todos los pares del dataset
      ↓
Multicolinealidad → por qué correlación alta entre features es un problema
      ↓
Reducción de dimensionalidad → qué hacer al respecto
      ↓
Límite del método → la correlación no lo ve todo
```

Este recorrido es la base del **EDA (Exploratory Data Analysis)**, la fase previa a la preparación del dataset (imputación, encoding, escalado) y al entrenamiento del modelo.
