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
