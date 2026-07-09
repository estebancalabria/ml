# Clase Dos - 9 de Julio 2026

# Repaso

* Setup del Entorno 
  * Python (Hay otros lenguajes como R)
    * Librerias
      * numpy
      * Pandas
      * MatplotLib
        * Seaborn
      * Scit-Learn
* ML
  * Tipos de Problemas
    * Regresion
      * Algoritmos
        * Regresion Lineal
      * Metricas
        * Error
    * Clasificacion
      * Algorimtos
        * Arboles Binario
      * Metricas
        * ??? <<< HOY
    * Overfitting
* Hola Mundo en ML
* ML en las organizaciones
* ML con el agente Analista de Copilot
  
---

# Librerias

* Pandas
  * https://pandas.pydata.org/
* MatplotLib
  * https://matplotlib.org/
  * Extension
    * Seaborn
      * https://seaborn.pydata.org/
* Sckit-learn
  * https://scikit-learn.org/stable/

---

# Entornos de ML

* Entornos On-Premise (in Company) especializados
 * Acceso remoto a notebboks
* Google Colab
 * https://colab.google/
* Proveedores Cloud
   * Azure
     * ML workspace
     * Studio
       * https://ml.azure.com/
   * AWS
     * ???
* Entornos Educativos
   * https://www.kaggle.com/

---

# Visualizacion de Datos con MatPlotLib

* Los graficos para ML son
  * plot (linea)
  * scatter (puntitos)
  * histograma (analizar la distribucion)
  * boxplot
  * heatmap (matriz de confusion y correlacion entre variables)

## Ejemplo Sencillo

```
import matplotlib.pyplot as plt

X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [2, 4, 9, 16, 25, 36, 49, 64, 81, 100]

plt.plot(X, y)

plt.show()
```

## Ejemplo con leyendas

```
import matplotlib.pyplot as plt

X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [2, 4, 9, 16, 25, 36, 49, 64, 81, 100]

plt.title("Gráfico de ejemplo")
plt.plot(X, y)
plt.xlabel("Eje X")
plt.ylabel("Eje Y")

plt.legend(["y = x^2"])
plt.show()
```

## Dos lineas

```
import matplotlib.pyplot as plt

X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [2, 4, 9, 16, 25, 36, 49, 64, 81, 100]
y_linea = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

plt.title("Gráfico de ejemplo")
plt.plot(X, y)
plt.plot(X, y_linea)
plt.xlabel("Eje X")
plt.ylabel("Eje Y")

plt.legend(["y = x^2", "y = x"])

plt.show()
```

<img width="425" height="341" alt="image" src="https://github.com/user-attachments/assets/06a9cf47-a483-4cc3-b3d1-af184e6c3061" />


## Histograma : Ejemplo Edades

```
edades = [20, 22, 25, 30, 35, 40, 45, 50, 55, 60, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 21]

plt.hist(edades, bins=5, color='blue', edgecolor='black')
plt.title("Histograma de edades")

plt.show()
```

<img width="407" height="322" alt="image" src="https://github.com/user-attachments/assets/e772591e-391a-4572-9838-030f00fc21d3" />


## Histograma : Ejemplo dados

```python
import random

#Generar un numero al azar entre 1 y 6
numero_azar = random.randint(1, 6)

#Generar un array de 1000 elementos donde cada elemento es la suma de dos valores al azar entre 1 y 6
suma_azar = [random.randint(1, 6) + random.randint(1, 6) for _ in range(1000)]

plt.hist(suma_azar, bins=11, color='green', edgecolor='black')
plt.title("Histograma de la suma de dos dados")
```

<img width="413" height="325" alt="image" src="https://github.com/user-attachments/assets/eaa0c5a5-0190-45f3-ae1f-595522ec8d33" />

## BoxPlot

```

curso_a = [2,  5,6,7,5,6,7, 10]
curso_b = [ 6, 7, 8, 9, 6, 7, 8, 9 ,6 ,7 ,8, 9]

plt.boxplot([curso_a, curso_b], labels=["Curso A", "Curso B"])

plt.show()
```
<img width="400" height="301" alt="image" src="https://github.com/user-attachments/assets/34b0be57-0609-408f-a250-73affb40f1a2" />

## Varios graficos y correlacion entre graficos

```
curso_a = [2,  5,6,7,5,6,7, 10]
curso_b = [ 6, 7, 8, 9, 6, 7, 8, 9 ,6 ,7 ,8, 9]

fig, ax = plt.subplots(1, 3, figsize=(12, 5))

ax[0].boxplot([curso_a, curso_b], labels=["Curso A", "Curso B"])

ax[1].hist(curso_a, bins=10, color='blue', edgecolor='black')
ax[1].set_title("Histograma Curso A")

ax[2].hist(curso_b, bins=10, color='green', edgecolor='black')
ax[2].set_title("Histograma Curso B")

plt.show()  
```

<img width="616" height="283" alt="image" src="https://github.com/user-attachments/assets/682b9703-ba2e-4d55-99b1-7af6cf2825a8" />

* Pedirle a la IA que saque conclusiones

```
Saca conclusiones de la correlacion entre datos con estos diagramas
```

* Ayuda a entender los datos

```
El Curso B muestra un rendimiento más consistente y homogéneo: todas las notas se concentran entre 6 y 9, mientras que en el Curso A hay alumnos desde 2 hasta 10.


La dispersión del Curso A es mucho mayor: el boxplot tiene un rango y una caja más amplios, lo que indica diferencias significativas entre los estudiantes y una mayor variabilidad en el rendimiento.


La mediana del Curso B es superior a la del Curso A: además de ser más homogéneo, el Curso B presenta un rendimiento típico más alto (≈ 7,5 frente a ≈ 6), sugiriendo mejores resultados generales del grupo.
```

## Grafico de Scatter (dispersion)

```
horas_estudio = [1, 2, 2, 4, 5, 6, 7, 8, 9, 9]
notas_examen = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

plt.scatter(horas_estudio, notas_examen)
plt.show()
```

<img width="402" height="307" alt="image" src="https://github.com/user-attachments/assets/7f06ffb6-6aa7-43b0-bb16-34591c6e633d" />

## Heatmap

```
import numpy as np

horas_estudio = [1, 2, 2, 4, 5, 6, 7, 8, 9, 9]
horas_sueno = [6, 7, 8, 9, 6, 7, 8, 9 ,6 ,7]
notas_examen = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

#Simulando el resultado de un ML
correlacion = np.corrcoef([horas_estudio, horas_sueno, notas_examen]);

plt.imshow(correlacion, cmap='coolwarm', interpolation='nearest')
plt.xticks([0, 1, 2], ['Horas de estudio', 'Horas de sueño', 'Notas examen'])
plt.yticks([0, 1, 2], ['Horas de estudio', 'Horas de sueño', 'Notas examen'])
plt.colorbar()

plt.show()
```

<img width="433" height="308" alt="image" src="https://github.com/user-attachments/assets/bac888b9-319a-4c29-9b8e-d34148c60fdd" />



### Resumen de metodos

* Para texto en el grafico
 * title
 * xlabel
 * ylabel
 * xticks
 * yticks
 * legend
* Graficos concretos
  * plot : graficos de Lineas
  * hist : histograma
  * boxplot : valor min, max, mediana
  * scatter
  * imshow
 * Finalmente
  * Show

---

# Kahoot 1

* https://kahoot.it/?pin=2887222&refer_method=link

---

# Proceso de ML

* Analizar la muestra de datos
  * Hacer un histograama para cada variable
  * Hacer un boxplot para cada variable (puede ser en el mismo grafico)
  * Tomar variables y probar su relacion
      * Regresion Lineal
      * Grafico de Scatter y plot

---

# Integrando Regresion Lineal con MatPlotLib

* Datos Originales
```
horas_estudio = [1, 2, 3, 4, 5, 6, 7, 8, 9, 9]
notas_examen = [54, 45, 55, 70, 70, 75, 65, 80, 85, 95]

plt.scatter(horas_estudio, notas_examen)
plt.show()
```

* Datos Originales
```
horas_estudio = [1, 2, 3, 4, 5, 6, 7, 8, 9, 9]
notas_examen = [54, 45, 55, 70, 70, 75, 65, 80, 85, 95]

plt.scatter(horas_estudio, notas_examen)
plt.show()
```

```
from sklearn.linear_model import LinearRegression

horas_estudio = [1, 2, 3, 4, 5, 6, 7, 8, 9, 9]
notas_examen = [54, 45, 55, 70, 70, 75, 65, 80, 85, 95]

# EN SKLEARN los features los pasamos de esta manera, como un array de arrays, por eso hacemos la transformación de X
#X = [ [1], [2], [3], [4], [5], [6], [7], [8], [9], [9] ]

X = [[x] for x in horas_estudio]
y = notas_examen

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

plt.scatter(horas_estudio, notas_examen)
plt.plot(horas_estudio, y_pred, color='red')
plt.show()
```

* Poder imprimit la formula

```
print("Coeficiente de regresión (pendiente):", model.coef_[0])
print("Intercepto:", model.intercept_)
print("Formula: y =", model.coef_[0], "* x +", model.intercept_)
```

---

# Segundo Kahoot (Break)

* https://kahoot.it/?pin=1125024&refer_method=link

---

# Clasificacion

* Predecir si estudiante aprueba o no dependiendo de sus horas de estudio y asistencias a clase

```python
from sklearn.linear_model import LogisticRegression

#Horas estudio, asistencia a clase
X = [
    [1 ,10],
    [2 ,6],
    [2 ,1],
    [3 ,9],
    [3 ,10],
    [4 ,8],
    [5 ,7],
    [5 ,10],
    [3 ,6],
    [9 ,5]
]

# 1, aprueba, 2 no aprueba
y = [0, 0, 0, 1, 1, 1, 0, 1, 0, 1]
```

* El ejemplo concreto

```
from sklearn.linear_model import LogisticRegression

#Horas estudio, asistencia a clase
X = [
    [1 ,10],
    [2 ,6],
    [2 ,1],
    [3 ,9],
    [3 ,10],
    [4 ,8],
    [5 ,7],
    [5 ,10],
    [3 ,6],
    [9 ,5]
]

# 1, aprueba, 2 no aprueba
y = [0, 0, 0, 1, 1, 1, 0, 1, 0, 1]

model = LogisticRegression()
model.fit(X, y)

nuevo_estudiante = [[5, 10]]

prediccion = model.predict(nuevo_estudiante)

print(prediccion)

if (prediccion[0] == 1):
    print("El nuevo estudiante aprueba")
else:
    print("El nuevo estudiante no aprueba")
```

> [!NOTE]
> Cambien los valores de el nuesvo estudiante de hoyras de estudio y asistencia para ver como varia la prediccion

# Metricas Clasificacion (Matriz de Confusion)

# Comparar Modelos

# Test/Train
