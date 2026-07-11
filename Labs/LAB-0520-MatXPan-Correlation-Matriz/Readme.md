# LAB-0520 · Matplotlib + Pandas: Matriz de Correlación

## Objetivo

Visualizar la matriz de correlación de tres variables (horas de estudio, nota y asistencia) como un **heatmap** con Matplotlib, y complementarlo con gráficos de dispersión (scatter) con la recta de regresión trazada, usando esta vez **pandas** y un **DataFrame** para el cálculo de correlación (sin NumPy), tal como se planteó en el LAB-0330 pero con otra librería base.

## En este laboratorio aprenderás...

- A organizar los datos en un `DataFrame` de pandas en lugar de arrays sueltos.
- A calcular la matriz de correlación con `DataFrame.corr()`, el método nativo de pandas.
- A generar un heatmap con Matplotlib a partir de esa matriz, y anotar cada celda con su valor.
- A graficar scatter plots con recta ajustada (calculada también con pandas, sin NumPy), tanto para el par correlacionado como para el que no lo está.

---

## Paso 1 — Preparar el entorno

```bash
pip install pandas matplotlib
```

```python
import pandas as pd
import matplotlib.pyplot as plt
```

---

## Paso 2 — Datos del laboratorio (mismos datos, en un DataFrame)

```python
df = pd.DataFrame({
    "Horas": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Nota": [4, 3.8, 6, 6, 5, 8, 8.0, 10, 8.3, 9.1],
    # A propósito, Asistencia NO sigue el mismo orden que Horas/Nota:
    # así queda sin correlación real, y la matriz muestra un caso con
    # correlación (Horas-Nota) y otro sin ella (Asistencia vs el resto).
    "Asistencia": [92, 55, 98, 60, 85, 65, 90, 58, 95, 70]
})

print(df)
```

**Qué cambia respecto al LAB-0330:** en vez de tener tres arrays de NumPy sueltos (`horas_A`, `notas_A`, `asistencia_A`), acá los datos viven juntos en un solo `DataFrame`, como columnas de una tabla. Esto es justamente el formato con el que vas a trabajar en datasets reales (un CSV, una consulta SQL, etc.).

---

## Paso 3 — Calcular la matriz de correlación con pandas

```python
matriz_corr = df.corr()
print(matriz_corr)
```

**Qué está pasando:** `DataFrame.corr()` calcula, por defecto, el coeficiente de correlación de Pearson entre cada par de columnas numéricas del DataFrame, y devuelve el resultado como otro `DataFrame` (no como un array de NumPy). Es el mismo cálculo que hicimos "a mano" en el LAB-ML-0110 y con `np.corrcoef()` en el LAB-0330 — acá lo hace pandas directamente, sin que nosotros llamemos a NumPy en ningún momento.

---

## Paso 4 — Construir el heatmap

```python
nombres_variables = matriz_corr.columns.tolist()

fig, ax = plt.subplots(figsize=(6, 5))

# Dibuja la matriz como imagen de color. vmin/vmax fijan la escala entre -1 y 1
im = ax.imshow(matriz_corr, cmap="coolwarm", vmin=-1, vmax=1)

# Ejes con los nombres de las variables (tomados directamente del DataFrame)
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
        valor = matriz_corr.iloc[i, j]
        ax.text(j, i, f"{valor:.2f}", ha="center", va="center", color="black")

ax.set_title("Matriz de Correlación: Horas, Nota, Asistencia")
plt.tight_layout()
plt.savefig("heatmap_correlacion_pandas.png", dpi=150)
plt.show()
```

**Qué está pasando en cada parte:**
- `imshow` acepta el `DataFrame` `matriz_corr` directamente (matplotlib lo interpreta como una tabla de valores), sin necesidad de convertirlo con NumPy.
- `nombres_variables` se obtiene de `matriz_corr.columns`, en vez de escribir una lista a mano — así el gráfico queda sincronizado con las columnas reales del DataFrame.
- `matriz_corr.iloc[i, j]` es la forma de pandas de acceder a una celda por posición (fila `i`, columna `j`), equivalente al `matriz[i, j]` que usábamos con arrays de NumPy.
- `cmap="coolwarm"`, `vmin=-1`, `vmax=1` cumplen la misma función que en el LAB-0330: fijar una escala de color comparable, de azul (correlación negativa) a rojo (correlación positiva).

**Cómo leer el resultado:**
- La **diagonal** siempre se ve del color más "caliente" extremo (rojo intenso), porque son correlaciones de 1.0 (cada variable consigo misma).
- La celda entre **Horas** y **Nota** debería verse roja/intensa: confirma la correlación lineal fuerte y positiva.
- Las celdas de **Asistencia** contra **Horas** y contra **Nota** deberían verse en tonos claros, cercanos al blanco: esa variable fue armada a propósito sin relación real, así que su color casi no se aleja del 0.

---

## Paso 5 — Scatter plot: caso CON correlación (Horas vs. Nota)

Para la recta ajustada, en vez de `np.polyfit` (que depende de NumPy), calculamos la pendiente y la ordenada al origen con los métodos propios de pandas: `.cov()`, `.var()` y `.mean()`.

```python
pendiente_hn = df["Horas"].cov(df["Nota"]) / df["Horas"].var()
ordenada_hn = df["Nota"].mean() - pendiente_hn * df["Horas"].mean()
recta_hn = pendiente_hn * df["Horas"] + ordenada_hn

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(df["Horas"], df["Nota"], color="steelblue", label="Datos reales")
ax.plot(df["Horas"], recta_hn, color="firebrick", label="Recta ajustada")
ax.set_xlabel("Horas de estudio")
ax.set_ylabel("Nota")
ax.set_title(f"Horas vs. Nota (r = {matriz_corr.loc['Horas', 'Nota']:.2f})")
ax.legend()
plt.tight_layout()
plt.savefig("scatter_horas_nota_pandas.png", dpi=150)
plt.show()
```

**Qué está pasando:** la fórmula de la recta de regresión simple es `pendiente = covarianza(X,Y) / varianza(X)` y `ordenada = media(Y) - pendiente * media(X)`. Es exactamente la misma matemática que usa `np.polyfit` por dentro, pero acá la armamos con los métodos de pandas (`.cov()`, `.var()`, `.mean()`), sin llamar a NumPy en ningún punto. Además, `matriz_corr.loc['Horas', 'Nota']` es la forma de pandas de acceder a una celda por **nombre** (en vez de por posición numérica como con `.iloc`).

**Qué deberías ver:** los puntos siguen de cerca la recta roja, sin desviarse demasiado. Eso es, visualmente, lo que significa un `r` cercano a 1.

---

## Paso 6 — Scatter plot: caso SIN correlación (Horas vs. Asistencia)

```python
pendiente_ha = df["Horas"].cov(df["Asistencia"]) / df["Horas"].var()
ordenada_ha = df["Asistencia"].mean() - pendiente_ha * df["Horas"].mean()
recta_ha = pendiente_ha * df["Horas"] + ordenada_ha

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(df["Horas"], df["Asistencia"], color="steelblue", label="Datos reales")
ax.plot(df["Horas"], recta_ha, color="firebrick", label="Recta ajustada")
ax.set_xlabel("Horas de estudio")
ax.set_ylabel("Asistencia")
ax.set_title(f"Horas vs. Asistencia (r = {matriz_corr.loc['Horas', 'Asistencia']:.2f})")
ax.legend()
plt.tight_layout()
plt.savefig("scatter_horas_asistencia_pandas.png", dpi=150)
plt.show()
```

**Qué deberías ver:** los puntos están dispersos alrededor de la recta sin seguir ningún patrón claro, y la recta ajustada queda casi plana (pendiente cercana a 0). El cálculo siempre te va a devolver una recta (la fórmula no sabe "negarse" a ajustar), pero esa recta no representa una relación real entre las variables.

---

## Qué conseguiste

- Organizaste los mismos datos del LAB-0330 en un `DataFrame` de pandas, el formato más habitual al trabajar con datos tabulares reales.
- Calculaste la matriz de correlación con `DataFrame.corr()`, sin usar NumPy en ningún momento del cálculo.
- Construiste el mismo heatmap que en el LAB-0330, pero leyendo los nombres de variables y los valores directamente desde el DataFrame (`.columns`, `.iloc`, `.loc`).
- Calculaste la recta de regresión "a mano" con métodos de pandas (`.cov()`, `.var()`, `.mean()`) en vez de `np.polyfit`, y la graficaste en dos scatter plots: uno para el par correlacionado y otro para el que no lo está.

---

## Código completo (copiar y pegar)

```python
import pandas as pd
import matplotlib.pyplot as plt

# --- Datos ---
df = pd.DataFrame({
    "Horas": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Nota": [4, 3.8, 6, 6, 5, 8, 8.0, 10, 8.3, 9.1],
    # A propósito, Asistencia NO sigue el mismo orden que Horas/Nota:
    # así queda sin correlación real, y la matriz muestra un caso con
    # correlación (Horas-Nota) y otro sin ella (Asistencia vs el resto).
    "Asistencia": [92, 55, 98, 60, 85, 65, 90, 58, 95, 70]
})

# --- Matriz de correlación con pandas ---
matriz_corr = df.corr()
print(matriz_corr)

nombres_variables = matriz_corr.columns.tolist()

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
        valor = matriz_corr.iloc[i, j]
        ax.text(j, i, f"{valor:.2f}", ha="center", va="center", color="black")
ax.set_title("Matriz de Correlación: Horas, Nota, Asistencia")
plt.tight_layout()
plt.savefig("heatmap_correlacion_pandas.png", dpi=150)
plt.show()

# --- Gráfico 2: Scatter CON correlación (Horas vs. Nota) ---
pendiente_hn = df["Horas"].cov(df["Nota"]) / df["Horas"].var()
ordenada_hn = df["Nota"].mean() - pendiente_hn * df["Horas"].mean()
recta_hn = pendiente_hn * df["Horas"] + ordenada_hn

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(df["Horas"], df["Nota"], color="steelblue", label="Datos reales")
ax.plot(df["Horas"], recta_hn, color="firebrick", label="Recta ajustada")
ax.set_xlabel("Horas de estudio")
ax.set_ylabel("Nota")
ax.set_title(f"Horas vs. Nota (r = {matriz_corr.loc['Horas', 'Nota']:.2f})")
ax.legend()
plt.tight_layout()
plt.savefig("scatter_horas_nota_pandas.png", dpi=150)
plt.show()

# --- Gráfico 3: Scatter SIN correlación (Horas vs. Asistencia) ---
pendiente_ha = df["Horas"].cov(df["Asistencia"]) / df["Horas"].var()
ordenada_ha = df["Asistencia"].mean() - pendiente_ha * df["Horas"].mean()
recta_ha = pendiente_ha * df["Horas"] + ordenada_ha

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(df["Horas"], df["Asistencia"], color="steelblue", label="Datos reales")
ax.plot(df["Horas"], recta_ha, color="firebrick", label="Recta ajustada")
ax.set_xlabel("Horas de estudio")
ax.set_ylabel("Asistencia")
ax.set_title(f"Horas vs. Asistencia (r = {matriz_corr.loc['Horas', 'Asistencia']:.2f})")
ax.legend()
plt.tight_layout()
plt.savefig("scatter_horas_asistencia_pandas.png", dpi=150)
plt.show()
```
