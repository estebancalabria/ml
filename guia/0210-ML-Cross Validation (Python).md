# Clase: Introducción a Cross Validation (Anexo con python)

## Objetivos

Al finalizar esta clase los estudiantes serán capaces de:

* Comprender por qué necesitamos comparar modelos.
* Entender la diferencia entre entrenamiento y evaluación.
* Comprender las limitaciones de una única división Train/Test.
* Entender intuitivamente qué es Cross Validation.
* Comprender qué significa un Fold.
* Diferenciar 4-Fold, 10-Fold y Leave-One-Out.
* Entender cómo Cross Validation ayuda a elegir modelos y parámetros.

## Secuencia

```text
Problema del corazón
        ↓
Tenemos varios algoritmos
        ↓
Necesitamos comparar
        ↓
Training Data
        ↓
Testing Data
        ↓
Problema de una sola división
        ↓
¿Qué bloque usar?
        ↓
Cross Validation
        ↓
Primer fold
        ↓
Segundo fold
        ↓
Tercer fold
        ↓
Cuarto fold
        ↓
Promediar resultados
        ↓
Elegir mejor modelo
        ↓
4-Fold
        ↓
10-Fold
        ↓
Leave One Out
        ↓
Tuning Parameters
        ↓
Conclusión
```

***

# Preparación Inicial

Durante toda la clase utilizaremos el dataset Breast Cancer incluido en Scikit-Learn.

## Ejecutar

```python
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)

print("Cantidad de pacientes:", len(X))
print("Cantidad de variables:", X.shape[1])
```

***

## ¿Qué representan estos datos?

```text
X = Características de los pacientes

y = Diagnóstico
    0 = maligno
    1 = benigno
```

***

# ¿Por qué necesitamos Cross Validation?

Supongamos que queremos construir un sistema para detectar enfermedades cardíacas.

Disponemos de información de muchos pacientes.

Para cada paciente conocemos:

* Dolor en el pecho
* Circulación sanguínea
* Edad
* Presión arterial
* Otros datos clínicos

Y además sabemos si tenía o no enfermedad cardíaca.

***

## Nuestro objetivo

Cuando llegue un paciente nuevo queremos responder:

```text
¿Tiene enfermedad cardíaca?
```

o

```text
¿No tiene enfermedad cardíaca?
```

Estamos ante un problema de:

# Clasificación

***

# Tenemos varios algoritmos

Podríamos utilizar:

* Regresión Logística
* KNN
* SVM
* Árboles de decisión
* Random Forest

y muchos más.

***

## Pregunta

¿Cuál elegimos?

```text
Logistic Regression
KNN
SVM
Random Forest
```

Todos parecen razonables.

¿Cómo sabemos cuál es mejor?

***

# Una idea equivocada

Podríamos entrenar todos los algoritmos usando todos los datos disponibles.

```text
████████████████████
Todos los pacientes
```

***

## Problema

Si utilizamos todos los datos para entrenar:

```text
No quedan datos para evaluar
```

Y entonces no sabremos cómo funcionarán con pacientes nuevos.

***

# La Regla de Oro

Nunca debemos evaluar un modelo usando los mismos datos con los que fue entrenado.

Porque eso mide memoria.

No mide capacidad de generalización.

***

## Recordatorio

Machine Learning no busca memorizar.

Busca predecir correctamente casos nuevos.

***

# Primera Solución: Train/Test Split

Dividamos los datos.

```text
████████████████░░░░

75% Entrenamiento
25% Prueba
```

***

## Hacer la división en Python

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

print("Train:", X_train.shape)
print("Test :", X_test.shape)
```

***

## Entrenamiento

```text
████████████████
```

Sirve para aprender.

***

## Testing

```text
░░░░
```

Sirve para evaluar.

***

# Ahora sí podemos comparar

Entrenamos varios modelos.

```text
Logistic Regression
KNN
SVM
```

***

## Ejecutar

```python
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

logistic = LogisticRegression(max_iter=10000)
knn = KNeighborsClassifier()
svm = SVC()

logistic.fit(X_train, y_train)
knn.fit(X_train, y_train)
svm.fit(X_train, y_train)
```

***

## Evaluar

```python
print("Logistic:", logistic.score(X_test, y_test))
print("KNN     :", knn.score(X_test, y_test))
print("SVM     :", svm.score(X_test, y_test))
```

***

## Pregunta

```text
¿Cuál obtuvo mejor accuracy?
```

***

# Pero aparece un problema

¿Por qué elegimos precisamente ese 25%?

¿Por qué no otro?

***

# Experimento

Probemos otra división.

## Ejecutar

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=7
)

modelo = LogisticRegression(max_iter=10000)

modelo.fit(X_train, y_train)

print(modelo.score(X_test, y_test))
```

***

Ahora probemos otra vez.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=15
)

modelo = LogisticRegression(max_iter=10000)

modelo.fit(X_train, y_train)

print(modelo.score(X_test, y_test))
```

***

## Observación

Los resultados pueden cambiar.

Entonces surge una pregunta importante:

> ¿Y si el resultado depende del bloque elegido?

***

# Nace Cross Validation

En lugar de probar un único bloque...

Probamos todos.

***

# 4-Fold Cross Validation

Dividimos los datos en cuatro partes.

```text
[1][2][3][4]
```

Cada parte recibe el nombre de:

# Fold

***

# Fold 1

Entrenamiento:

```text
[1][2][3]
```

Prueba:

```text
[4]
```

***

# Fold 2

Entrenamiento:

```text
[1][2][4]
```

Prueba:

```text
[3]
```

***

# Fold 3

Entrenamiento:

```text
[1][3][4]
```

Prueba:

```text
[2]
```

***

# Fold 4

Entrenamiento:

```text
[2][3][4]
```

Prueba:

```text
[1]
```

***

# Realizando Cross Validation

## Ejecutar

```python
from sklearn.model_selection import cross_val_score

modelo = LogisticRegression(max_iter=10000)

scores = cross_val_score(
    modelo,
    X,
    y,
    cv=4
)

print(scores)
```

***

Ejemplo de salida:

```text
[0.94 0.95 0.96 0.93]
```

***

Cada número corresponde a un Fold distinto.

***

# Resultado Final

Calculamos el promedio.

## Ejecutar

```python
print(scores.mean())
```

***

También podemos ver la variabilidad.

```python
print(scores.std())
```

***

## Interpretación

```text
mean() = rendimiento medio

std() = cuánto varían los resultados
```

***

# ¿Qué hemos conseguido?

Cada observación fue utilizada:

* Para entrenar
* Para probar

Pero nunca ambas cosas al mismo tiempo.

***

Y además:

```text
Todos los bloques tuvieron
su oportunidad de ser Testing
```

***

# Comparando Modelos

Ahora sí podemos comparar algoritmos de forma más justa.

## Ejecutar

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

modelos = {
    "Logistic": LogisticRegression(max_iter=10000),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC()
}
```

***

```python
for nombre, modelo in modelos.items():

    scores = cross_val_score(
        modelo,
        X,
        y,
        cv=10
    )

    print(
        nombre,
        "=>",
        round(scores.mean(), 4)
    )
```

***

## Preguntas

```text
¿Cuál obtuvo mejor accuracy media?

¿Era también el mejor con un único Train/Test Split?
```

***

# ¿Siempre son 4 Folds?

No.

El número de folds puede cambiar.

***

# 10-Fold Cross Validation

Es la variante más utilizada.

```text
[1][2][3][4][5]
[6][7][8][9][10]
```

***

## Ejecutar

```python
scores = cross_val_score(
    LogisticRegression(max_iter=10000),
    X,
    y,
    cv=10
)

print(scores)
print(scores.mean())
```

***

## Regla práctica

Cuando alguien dice:

```text
Cross Validation
```

Muchas veces se refiere a:

```text
10-Fold Cross Validation
```

***

# Leave One Out Cross Validation

Llevemos la idea al extremo.

Cada paciente es un Fold.

```text
Paciente 1
Paciente 2
Paciente 3
...
Paciente N
```

***

## Ejecutar

```python
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()

scores = cross_val_score(
    LogisticRegression(max_iter=10000),
    X,
    y,
    cv=loo
)

print("Número de folds:", len(scores))
```

***

## Pregunta

```text
¿Cuántos modelos se entrenaron?
```

***

## Ventaja

Aprovecha prácticamente todos los datos para entrenar.

***

## Desventaja

Puede tardar mucho tiempo.

***

# ¿Para Qué Más Sirve Cross Validation?

No solo sirve para comparar modelos.

También sirve para ajustar parámetros.

***

# El Caso de KNN

KNN tiene un parámetro importante.

```text
n_neighbors
```

o simplemente:

```text
K
```

***

## ¿Qué valor elegimos?

Probemos varios.

***

## Ejecutar

```python
for k in [1, 3, 5, 7, 9]:

    modelo = KNeighborsClassifier(
        n_neighbors=k
    )

    scores = cross_val_score(
        modelo,
        X,
        y,
        cv=10
    )

    print(
        "K =", k,
        "Accuracy =", round(scores.mean(), 4)
    )
```

***

## Interpretación

Cross Validation permite comparar:

```text
K=1

K=3

K=5

K=7

K=9
```

y elegir el mejor valor.

***

# Idea Fundamental

Cross Validation responde una pregunta muy importante:

> ¿Cómo creemos que funcionará el modelo con datos que todavía no hemos visto?

***

# Lo Que NO Hace

Cross Validation no garantiza perfección.

No garantiza que el modelo sea correcto.

No garantiza que el resultado futuro sea idéntico.

***

Lo que hace es proporcionar una estimación mucho más confiable del rendimiento real del modelo.

***

# Resumen Final

Cross Validation permite evaluar modelos de forma más robusta.

En lugar de depender de una única división Train/Test:

```text
Probamos muchas divisiones
```

Cada bloque se convierte en Testing una vez.

Los resultados se promedian.

Esto permite:

* Comparar algoritmos.
* Elegir el mejor modelo.
* Ajustar hiperparámetros.
* Reducir el riesgo de tomar decisiones basadas en una división poco representativa.

***

# Conceptos de Python Aprendidos

Durante esta clase utilizamos:

```python
train_test_split()
```

Para dividir datos en entrenamiento y prueba.

***

```python
fit()
```

Para entrenar modelos.

***

```python
score()
```

Para evaluar modelos.

***

```python
cross_val_score()
```

Para realizar Cross Validation.

***

```python
LeaveOneOut()
```

Para Leave-One-Out Cross Validation.

***

# Mensaje de cierre

> El objetivo de Cross Validation no es obtener la puntuación más alta.
>
> Su objetivo es obtener una estimación más realista de cómo funcionará el modelo cuando aparezcan datos nuevos. 🚀
