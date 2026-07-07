# Clase: Introducción a Cross Validation

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
Random Forest
```

***

Y obtenemos resultados.

```text
Modelo A → 82%

Modelo B → 84%

Modelo C → 88%

Modelo D → 90%
```

Parecería que gana el Modelo D.

***

# Pero aparece un problema

¿Por qué elegimos precisamente ese 25%?

¿Por qué no otro?

***

Imaginemos los datos divididos en cuatro bloques.

```text
[1][2][3][4]
```

***

Podríamos usar:

```text
[1][2][3] Train

[4] Test
```

***

Pero también podríamos usar:

```text
[2][3][4] Train

[1] Test
```

***

O cualquier otro bloque.

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

Supongamos que obtenemos:

```text
90%
```

Guardamos el resultado.

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

Resultado:

```text
86%
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

Resultado:

```text
88%
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

Resultado:

```text
92%
```

***

# Resultado Final

Ahora calculamos el promedio.

```text
90
86
88
92
----
89
```

Promedio:

```text
89%
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

Supongamos:

```text
Logistic Regression → 85%

KNN → 87%

Random Forest → 88%

SVM → 91%
```

***

Ahora la comparación es mucho más fiable.

Porque el resultado no depende de una única división de los datos.

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

Existe un equilibrio razonable entre:

* velocidad
* precisión

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

## ¿Qué ocurre?

Entrenamos con todos menos uno.

Y probamos con el que quedó fuera.

Luego repetimos el proceso para cada observación.

***

Ejemplo:

```text
99 pacientes → Entrenamiento

1 paciente → Testing
```

***

## Ventaja

Aprovecha prácticamente todos los datos para entrenar.

***

## Desventaja

Puede tardar muchísimo tiempo.

***

# ¿Para Qué Más Sirve Cross Validation?

No solo sirve para comparar modelos.

También sirve para ajustar parámetros.

***

# El Caso de Ridge Regression

Ridge Regression tiene un parámetro llamado:

```text
λ (lambda)
```

***

El algoritmo no aprende automáticamente ese valor.

Debemos elegirlo.

***

Podemos probar:

```text
λ = 0.01

λ = 0.1

λ = 1

λ = 10
```

***

Y utilizar Cross Validation para decidir cuál funciona mejor.

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

## Mensaje de cierre

> El objetivo de Cross Validation no es obtener la puntuación más alta.
>
> Su objetivo es obtener una estimación más realista de cómo funcionará el modelo cuando aparezcan datos nuevos. 🚀

Pedagógicamente, Esteban, esta clase encaja perfectamente después de tu clase de **Training Data / Testing Data / Generalización**. De hecho, yo recuperaría explícitamente el ejemplo de la **línea negra vs curva verde** y diría:

> "La semana pasada aprendimos a separar Training y Testing. Hoy veremos que una sola separación puede engañarnos, y cómo Cross Validation resuelve ese problema."

Así queda una transición muy natural dentro del recorrido del curso.
