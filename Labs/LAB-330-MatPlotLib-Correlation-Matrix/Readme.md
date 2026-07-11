# LAB-330 · Matplotlib: Heatmap de Matriz de Correlación

## Objetivo

Visualizar la matriz de correlación de tres variables (horas de estudio, nota y asistencia) como un **heatmap** con Matplotlib, y complementarlo con gráficos de dispersión (scatter) con la recta de regresión trazada, para poder identificar tanto numérica como visualmente qué pares de variables están relacionados linealmente y cuáles no. Reutiliza los mismos datos del LAB-ML-0110.

## En este laboratorio aprenderás...

- A generar un heatmap con `matplotlib` a partir de una matriz de correlación.
- A anotar cada celda del heatmap con su valor numérico para no depender solo del color.
- A graficar un scatter plot con la recta de regresión ajustada, tanto para un par de variables correlacionado como para uno que no lo está.
- A conectar lo que ves en el heatmap y en los scatter plots con los coeficientes que ya calculaste manualmente en el LAB-ML-0110.

---

## Paso 1 — Preparar el entorno

```bash
pip install numpy matplotlib
```

```python
import numpy as np
import matplotlib.pyplot as plt
```

---

## Paso 2 — Datos del laboratorio

```python
horas_A = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
notas_A = np.array([4, 3.8, 6, 6, 5, 8, 8.0, 10, 8.3, 9.1])

# A propósito, asistencia NO sigue el mismo orden que horas/nota:
# así queda sin correlación real, y la matriz muestra un caso con
# correlación (horas-nota) y otro sin ella (asistencia vs el resto).
asistencia_A = np.array([92, 55, 98, 60, 85, 65, 90, 58, 95, 70])

nombres_variables = ["Horas", "Nota", "Asistencia"]
datos_A = np.array([horas_A, notas_A, asistencia_A])
```

Nota: `notas_A` tiene ahora un poco más de ruido que en labs anteriores (no es una recta perfecta), lo cual es más realista y sigue mostrando correlación lineal fuerte con `horas_A`.

---

## Paso 3 — Calcular la matriz de correlación

Mismo paso que ya hicimos en el LAB-ML-0110, ahora es el punto de partida para graficar:

```python
matriz_corr = np.corrcoef(datos_A)
print(matriz_corr)
```

---

## Paso 4 — Construir el heatmap

```python
fig, ax = plt.subplots(figsize=(6, 5))

# Dibuja la matriz como imagen de color. vmin/vmax fijan la escala entre -1 y 1
im = ax.imshow(matriz_corr, cmap="coolwarm", vmin=-1, vmax=1)

# Ejes con los nombres de las variables
ax.set_xticks(range(len(nombres_variables)))
ax.set_yticks(range(len(nombres_variables)))
ax.set_xticklabels(nombres_variables)
ax.set_yticklabels(nombres_variables)

# Barra de color lateral que indica la escala -1 a 1
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Coeficiente de correlación (r)")

# Anotar cada celda con su valor numérico
for i in range(len(nombres_variables)):
    for j in range(len(nombres_variables)):
        valor = matriz_corr[i, j]
        ax.text(j, i, f"{valor:.2f}", ha="center", va="center", color="black")

ax.set_title("Matriz de Correlación: Horas, Nota, Asistencia")
plt.tight_layout()
plt.savefig("heatmap_correlacion.png", dpi=150)
plt.show()
```

**Qué está pasando en cada parte:**
- `imshow` pinta la matriz como una grilla de colores, donde el color de cada celda depende de su valor numérico.
- `cmap="coolwarm"` es un mapa de color que va de azul (frío, correlaciones negativas) a rojo (caliente, correlaciones positivas), pasando por blanco cerca del 0.
- `vmin=-1, vmax=1` fija la escala de color siempre entre -1 y 1, así el heatmap es comparable entre distintos datasets.
- El doble `for` agrega el número exacto arriba de cada celda, porque el color solo da una intuición — el valor preciso hace falta para no adivinar.

**Cómo leer el resultado:**
- La **diagonal** siempre se ve del color más "caliente" extremo (rojo intenso), porque son correlaciones de 1.0 (cada variable consigo misma).
- La celda entre **Horas** y **Nota** debería verse roja/intensa: confirma la correlación lineal fuerte y positiva.
- Las celdas de **Asistencia** contra **Horas** y contra **Nota** deberían verse en tonos claros, cercanos al blanco: esa variable fue armada a propósito sin relación real, así que su color casi no se aleja del 0.

---

## Paso 5 — Scatter plot: caso CON correlación (Horas vs. Nota)

El heatmap te da el número resumido, pero el scatter con la recta ajustada te muestra *cómo* se ve esa correlación en los datos crudos.

```python
# Ajuste de una recta (grado 1 = lineal): devuelve pendiente y ordenada al origen
pendiente_hn, ordenada_hn = np.polyfit(horas_A, notas_A, 1)
recta_hn = pendiente_hn * horas_A + ordenada_hn

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(horas_A, notas_A, color="steelblue", label="Datos reales")
ax.plot(horas_A, recta_hn, color="firebrick", label="Recta ajustada")
ax.set_xlabel("Horas de estudio")
ax.set_ylabel("Nota")
ax.set_title(f"Horas vs. Nota (r = {matriz_corr[0,1]:.2f})")
ax.legend()
plt.tight_layout()
plt.savefig("scatter_horas_nota.png", dpi=150)
plt.show()
```

**Qué deberías ver:** los puntos siguen de cerca la recta roja, sin desviarse demasiado. Eso es, visualmente, lo que significa un `r` cercano a 1.

---

## Paso 6 — Scatter plot: caso SIN correlación (Horas vs. Asistencia)

```python
pendiente_ha, ordenada_ha = np.polyfit(horas_A, asistencia_A, 1)
recta_ha = pendiente_ha * horas_A + ordenada_ha

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(horas_A, asistencia_A, color="steelblue", label="Datos reales")
ax.plot(horas_A, recta_ha, color="firebrick", label="Recta ajustada")
ax.set_xlabel("Horas de estudio")
ax.set_ylabel("Asistencia")
ax.set_title(f"Horas vs. Asistencia (r = {matriz_corr[0,2]:.2f})")
ax.legend()
plt.tight_layout()
plt.savefig("scatter_horas_asistencia.png", dpi=150)
plt.show()
```

**Qué deberías ver:** los puntos están dispersos alrededor de la recta sin seguir ningún patrón claro, y la recta ajustada queda casi plana (pendiente cercana a 0). Eso es lo que significa, visualmente, un `r` cercano a 0: aunque `np.polyfit` siempre te va a devolver una recta (el algoritmo no sabe "negarse"), esa recta no representa una relación real entre las variables.

---

## Qué conseguiste

- Construiste un heatmap con Matplotlib a partir de una matriz de correlación de tres variables.
- Aprendiste a fijar la escala de color (`vmin`/`vmax`) para que el heatmap sea interpretable y comparable entre distintos datasets.
- Graficaste dos scatter plots con recta ajustada (`np.polyfit`), uno para el par correlacionado y otro para el par sin relación, viendo en los datos crudos lo que el coeficiente `r` resume en un número.
- Conectaste tres formas de ver lo mismo: la matriz numérica, el heatmap y el scatter plot — cerrando el recorrido que empezaste en el LAB-0610 (regresión y R²) y seguiste en el LAB-ML-0110 (matriz de correlación con NumPy).

---

## Código completo (copiar y pegar)

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Datos ---
horas_A = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
notas_A = np.array([4, 3.8, 6, 6, 5, 8, 8.0, 10, 8.3, 9.1])

# A propósito, asistencia NO sigue el mismo orden que horas/nota:
# así queda sin correlación real, y la matriz muestra un caso con
# correlación (horas-nota) y otro sin ella (asistencia vs el resto).
asistencia_A = np.array([92, 55, 98, 60, 85, 65, 90, 58, 95, 70])

nombres_variables = ["Horas", "Nota", "Asistencia"]
datos_A = np.array([horas_A, notas_A, asistencia_A])

# --- Matriz de correlación ---
matriz_corr = np.corrcoef(datos_A)
print(matriz_corr)

# --- Gráfico 1: Heatmap de la matriz de correlación ---
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(matriz_corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(nombres_variables)))
ax.set_yticks(range(len(nombres_variables)))
ax.set_xticklabels(nombres_variables)
ax.set_yticklabels(nombres_variables)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Coeficiente de correlación (r)")
for i in range(len(nombres_variables)):
    for j in range(len(nombres_variables)):
        valor = matriz_corr[i, j]
        ax.text(j, i, f"{valor:.2f}", ha="center", va="center", color="black")
ax.set_title("Matriz de Correlación: Horas, Nota, Asistencia")
plt.tight_layout()
plt.savefig("heatmap_correlacion.png", dpi=150)
plt.show()

# --- Gráfico 2: Scatter CON correlación (Horas vs. Nota) ---
pendiente_hn, ordenada_hn = np.polyfit(horas_A, notas_A, 1)
recta_hn = pendiente_hn * horas_A + ordenada_hn

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(horas_A, notas_A, color="steelblue", label="Datos reales")
ax.plot(horas_A, recta_hn, color="firebrick", label="Recta ajustada")
ax.set_xlabel("Horas de estudio")
ax.set_ylabel("Nota")
ax.set_title(f"Horas vs. Nota (r = {matriz_corr[0,1]:.2f})")
ax.legend()
plt.tight_layout()
plt.savefig("scatter_horas_nota.png", dpi=150)
plt.show()

# --- Gráfico 3: Scatter SIN correlación (Horas vs. Asistencia) ---
pendiente_ha, ordenada_ha = np.polyfit(horas_A, asistencia_A, 1)
recta_ha = pendiente_ha * horas_A + ordenada_ha

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(horas_A, asistencia_A, color="steelblue", label="Datos reales")
ax.plot(horas_A, recta_ha, color="firebrick", label="Recta ajustada")
ax.set_xlabel("Horas de estudio")
ax.set_ylabel("Asistencia")
ax.set_title(f"Horas vs. Asistencia (r = {matriz_corr[0,2]:.2f})")
ax.legend()
plt.tight_layout()
plt.savefig("scatter_horas_asistencia.png", dpi=150)
plt.show()
```
