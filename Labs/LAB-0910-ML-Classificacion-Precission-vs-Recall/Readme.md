# Laboratorio: Precision y Recall cuando el Accuracy no alcanza

## Objetivos

- Entender por qué el accuracy puede ser engañoso en datasets desbalanceados.
- Calcular precision y recall a partir de predicciones hardcodeadas (sin entrenar ningún modelo).
- Comparar dos "modelos" con el mismo accuracy pero comportamientos opuestos.
- Decidir qué métrica priorizar según el costo de negocio de cada tipo de error.

## Contexto

Detección de fraude en transacciones. Tenemos 20 transacciones, de las cuales solo 4 son fraude real (clase minoritaria = 1). Este desbalance (16 vs 4) es justamente el escenario donde el accuracy puede esconder problemas serios.

En vez de entrenar un modelo, hardcodeamos `y_true` y dos vectores de predicciones (`y_pred_A` y `y_pred_B`) que simulan dos estrategias distintas.

```python
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

# Real: 16 transacciones normales (0) y 4 fraudulentas (1)
y_true = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1])

# Modelo A: "conservador" -> solo marca fraude cuando está muy seguro
y_pred_A = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0])

# Modelo B: "alarmista" -> prefiere marcar de más antes que dejar pasar fraude
y_pred_B = np.array([0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1,1,1,0])
```

## Métricas

```python
for nombre, y_pred in [("Modelo A", y_pred_A), ("Modelo B", y_pred_B)]:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"--- {nombre} ---")
    print("Accuracy: ", round(acc, 2))
    print("Precision:", round(prec, 2))
    print("Recall:   ", round(rec, 2))
    print("Matriz de confusión:\n", cm)
    print()
```

**Salida esperada:**

```
--- Modelo A ---
Accuracy:  0.85
Precision: 1.0
Recall:    0.25
Matriz de confusión:
 [[16  0]
 [ 3  1]]

--- Modelo B ---
Accuracy:  0.85
Precision: 0.6
Recall:    0.75
Matriz de confusión:
 [[14  2]
 [ 1  3]]
```

Los dos modelos tienen **exactamente el mismo accuracy (0.85)**. Si solo miráramos esa métrica, parecerían equivalentes. Pero fallan de maneras completamente distintas.

## Consignas (responder por escrito)

1. ¿Por qué el accuracy es igual en ambos modelos si sus matrices de confusión son tan diferentes?

>   **Respuesta:** Porque el accuracy solo cuenta el total de aciertos sobre el total de casos, sin distinguir qué tipo de error se cometió. Modelo A acierta mucho en la clase mayoritaria (no-fraude) y falla en la minoritaria (fraude), mientras que Modelo B distribuye los errores de otra forma. Como la clase 0 domina el dataset (16 de 20), ambos modelos "se apoyan" en acertar ahí y el accuracy termina pareciéndose, aunque el comportamiento frente al fraude real sea opuesto.

---
2. ¿Qué modelo tiene mejor precision y qué significa eso en este contexto?

   **Respuesta:** Modelo A tiene precision = 1.0: de todas las transacciones que marcó como fraude, el 100% realmente lo eran (no tuvo falsos positivos). Esto significa que cuando este modelo levanta una alerta, es muy confiable, pero a costa de dejar pasar fraudes reales sin detectar.

---
3. ¿Qué modelo tiene mejor recall y qué significa eso en este contexto? 

>   **Respuesta:** Modelo B tiene recall = 0.75: detectó 3 de los 4 fraudes reales. Esto significa que es mejor "atrapando" fraude, aunque para lograrlo también marca como sospechosas 2 transacciones legítimas (falsos positivos).

---
4. Si el banco pierde mucho dinero por cada fraude no detectado, pero bloquear una transacción legítima solo genera una molestia menor al cliente, ¿qué modelo elegís y por qué?

>   **Respuesta:** Modelo B, porque prioriza el recall. El costo de un falso negativo (fraude no detectado = pérdida de dinero) es mucho mayor que el costo de un falso positivo (cliente molesto por una alerta de más). En este escenario conviene tolerar más falsas alarmas a cambio de detectar más fraude real.

---
5. Si en cambio bloquear a un cliente legítimo le genera al banco un problema serio de reputación o pérdida de ese cliente, ¿qué modelo elegís?

 >  **Respuesta:** Modelo A, porque prioriza la precision. Acá el costo de un falso positivo es alto, entonces conviene un modelo que solo marque fraude cuando está muy seguro, aunque eso implique dejar pasar algunos casos reales.

## Conclusiones esperadas

- Dos modelos pueden tener el mismo accuracy y ser completamente distintos en la práctica.
- Precision responde a la pregunta: "de lo que marqué como positivo, ¿cuánto era realmente positivo?".
- Recall responde a la pregunta: "de todo lo que era realmente positivo, ¿cuánto logré detectar?".
- No existe una métrica "mejor" en abstracto: la elección depende del costo real de cada tipo de error en el problema de negocio.

## Código completo

```python
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

y_true = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1])

y_pred_A = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0])
y_pred_B = np.array([0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1,1,1,0])

for nombre, y_pred in [("Modelo A", y_pred_A), ("Modelo B", y_pred_B)]:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"--- {nombre} ---")
    print("Accuracy: ", round(acc, 2))
    print("Precision:", round(prec, 2))
    print("Recall:   ", round(rec, 2))
    print("Matriz de confusión:\n", cm)
    print()
```
