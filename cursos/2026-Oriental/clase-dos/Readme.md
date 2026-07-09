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

## Histograma : Ejemplo 1

```
edades = [20, 22, 25, 30, 35, 40, 45, 50, 55, 60, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 21]

plt.hist(edades, bins=5, color='blue', edgecolor='black')
plt.title("Histograma de edades")

plt.show()
```

### Resumen de metodos

* Para texto en el grafico
 * title
 * xlabel
 * ylabel
 * legend
* Graficos concretos
  * plot : graficos de Lineas
 * Finalmente
  * Show
# Clasificacion

# Metricas Clasificacion

# Comparar Modelos

# Test/Train
