# Clase Uno - 7 de Julio del 2026

# RoadMap

* Fundatmentos de ML
    * Que es
    * Como se trabaja en...
    * Aelinacion con el negocio
  * Primer Hola Mundo en ML
  * Liberias fundamentales de ML que vamos a usar
* Visualizacion de datos
   * MatPlotLib
   * Lo vamos a utilizar para poder entender los datos
   * Entender lo que me quiere decir la IA
* Trabajar con datos Estructurados (Tabulares)
   * Pandas
   * Nympy
   * Es un libreria para trabajar con datos
   * Analogia con SQL
      * El pandas es para ML, lo que el SQL es para Base de datos
      * Un DataFrame es en pandas para ML, lo que una Tabla es en una base de datos
* Machine Learning
   * Entrenar Modelos
   * Sckit-Learn
   * Entender el pipeline o forma de trabajo de ML en la practica
   * Primeras experieciencias entrenando modelos de ML
* Adv Machine Learning
   * Conocer el catalogo de algoritmos de Machine Learning
   * Comparar entre distintos algoritmos cual aplicar en cada caso
   * Metricas y graficos para entender la eficacia de los distintos algoritmos
   * Probar la efectividad de un modelo

# Presentacion

* Esteban Cesar Calabria
* Microsoft Certified Trainner
   * https://www.instagram.com/mct.esteban.calabria/
   * https://www.linkedin.com/in/esteban-calabria-7a44401a/

# Setup del entono

* Tenemos el VSCode
* Tenemos la extension Jypyter
* Creamos un archivo con extension ipynb
    * DEberia verse otra vista distinta donde temos dos botones +code +markdown
* Agregar una celda de codigo

```
print("Hola mundo")
```

* Presionar el boton "Play" que aparece al lado de la celda
* La primera vez me aparece un popup para elegir la version, elijo la que viene por defecto
* Luego ejecuto nuevamente

## Hacer nuestro primer grafico con MatPlotLib

```
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

plt.plot(x, y) 
plt.show()
```

# Fundamentos de Machine Learning

* Un arbol de decision utilizado para tareas de Clasificacion
  
<img width="938" height="562" alt="image" src="https://github.com/user-attachments/assets/956e7eec-dbf7-4b4e-80f8-582661d8673d" />

* Un ejemplo que hice con copilot
  
```python
print("=== Sistema de aprobación ===")

edad = int(input("Ingrese su edad: "))

if edad > 18:
    trabajo = input("¿Tiene trabajo? (si/no): ").lower()

    if trabajo == "si":
        print("Resultado: APROBADO")
    else:
        print("Resultado: RECHAZADO")
else:
    print("Resultado: RECHAZADO")
```

* Otros problemas de Clasificacion
   * Spam / No Spam
   * Le gusta / No le gusta
   * Tiene Diabetes/ No tiene Diabetes
   * Merlot / Cabenet / Syrah

* Un grafico para calcular una Regresion

<img width="1309" height="609" alt="image" src="https://github.com/user-attachments/assets/7235058e-4c4a-495d-b1b3-cc1c82259271" />

* Le pedi a copilot que me lo grafique con matplotlib

```
Me podes hacer este grafico con la libreria MatPlot Lib mostrando 4 puntos de los alumnos la nota que se sacaron y las horas de estudio y una recta roja aproximada que represente la regresion lineal. El grafico que te pase tiene solo 3 puntos. Agregar uno mas
```

* Me genero este codigo

```python
import matplotlib.pyplot as plt
import numpy as np

# Datos de ejemplo
horas_estudio = np.array([2, 4, 6, 8])
nota = np.array([3, 3, 8, 9])

# Calcular la recta de regresión
coeficientes = np.polyfit(horas_estudio, nota, 1)
pendiente, intercepto = coeficientes

# Generar puntos para la recta
x_recta = np.linspace(1, 9, 100)
y_recta = pendiente * x_recta + intercepto

# Gráfico
plt.figure(figsize=(8, 5))

# Puntos de alumnos
plt.scatter(
    horas_estudio,
    nota,
    color="blue",
    s=100,
    label="Alumnos"
)

# Etiquetas de cada punto
for i in range(len(horas_estudio)):
    plt.annotate(
        f"Alumno {i+1}",
        (horas_estudio[i], nota[i]),
        xytext=(5, 5),
        textcoords="offset points"
    )

# Recta de regresión
plt.plot(
    x_recta,
    y_recta,
    color="red",
    linewidth=2,
    label="Regresión lineal"
)

plt.title("Horas de estudio vs Nota obtenida")
plt.xlabel("Horas de estudio")
plt.ylabel("Nota")
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.grid(True, alpha=0.3)
plt.legend()

plt.show()
```
* Miremos este grafico

<img width="1309" height="609" alt="image" src="https://github.com/user-attachments/assets/3fe94224-2fff-4371-9d18-9baed02485a1" />

* Vamos a suponer que entrenamos a un algotirmo de ML para que me haga una formula para predecir el resultado en funcion de las horas de estudio
   * Opcion 1 (Roja) - Regresion Lineal - Apromaximacion
   * Opcion 2 (Azul) - Modelo Mas complejo - Funciona perfecto para los datos de prueba

* Variables
   * Horas de estudio (Feature)
   * Nota (Label) <- Variable que quiero predecir
 
* Conceptos
   * Datos de Entrenamiento (Puntos Negros)
   * Datos de Prueba / Nuevos (Puntos Verdes)
 
* Algoritmo de ML al que le damos los datos conocidos y en este caso nos devuelve un formula para relacionar ambas variables
   * Algoritmo 1 : Regresion Lineal (aproxima todo a un recta)
   * Algotimo 2: Genera un funcion de aproximacion mas compleja que pasa por todos los datos de entrenamiento

* (Datos de entenamiento) -> (Algotimo de ML 1)  ->  (Formula lineal que relaciona las variables) (Modelo 1)
                          -> (Algotimo de ML 2)  ->  (Formula compleja que relaciona las variables) (Modelo 2)

* Datos de Prueba
   * Apare la idea de Error. La distancia entre lo que tendria que dar segun el modelo y la realidad
 
> [!NOTE]
> A pesar de que la formula compleja predecia perfectamente los datos de entrenamiento, al momento de aplicarla con datos reales la formula lineal era mas generalizable, me daba mejores predicciones a la larga

* Overfitting : Cuando el modelo aprende bien los datos reales pero no generaliza bien con datos nuevos.

## Vamos a poner en prueba nuestros conocimientos

> https://kahoot.it/?pin=5875495&refer_method=link

## Hola Mundo ML

```python
from sklearn.linear_model import LinearRegression

X = [ 
        [1], 
        [2], 
        [3], 
        [4], 
        [5] 
    ]
y = [ 2, 4, 6, 8 , 10 ]

model = LinearRegression()

model.fit(X, y)
```

* Luego usamos el modelo para hacer una prediccion

```
prediccion = model.predict([[10]])
print(f"La predicción para 10 es: {prediccion[0]}")
```

## Otro ejemplo con valores mas realistas

```python
X = [ 
        [0.8], 
        [2.2], 
        [2.9], 
        [4.1], 
        [5.2] 
    ]
y = [ 2, 4, 6, 8 , 10 ]

model = LinearRegression()

model.fit(X, y)
```

* Lo probamos

```
prediccion = model.predict([[10]])

print(f"La predicción para 10 es: {prediccion[0]}")
```

* Esto me devuelve

```
La predicción para 10 es: 18.915712799167526
```

## Ejemplo con mas de dos features

```python
X = [ 
        [1, 1], 
        [2, 3], 
        [3, 3], 
        [4, 2], 
        [5, 1] 
    ]
y = [ 1, 6, 9, 8 , 5 ]

model = LinearRegression()

model.fit(X, y)

```

## Glosario

* Tipos de Problema ML
   * Clasificacion
      * Algoritmos
         * ArbolDeDecision
         * ...
   * Regresion
      * Algoritmos
         * RegresionLineal
         * ...
* Datos
   * Datos de Entrenamiento
   * Datos de Prueba/Testing

* Tipos de Variables
   * Features (X) (X mayuscula): Las variables que usamos para hacer predicciones
   * Labels (y) (y minuscula): Variables que queremos predecir
   
* Algorimo de ML -> Modelo (Ej (RegresionLineal -> FormulaMAtematica)

* Conceptos
   * Error : LA diferencia entre la prediccion del modelo y lo que tendria que dar
   * Overfittin : Cuando se memorizan los datos de entreamineto pero no generaliza bien
  
