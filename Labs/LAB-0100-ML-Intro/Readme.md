# LAB-0600 – Machine Learning: Hola Mundo

## Objetivo

Ver en pocas líneas qué es Machine Learning: una computadora que aprende un patrón a partir de ejemplos, sin que se lo programemos a mano.

## Requisitos previos

```
pip install scikit-learn
```

## El ejemplo

Le vamos a dar a la computadora algunos ejemplos de números y su doble, sin decirle la fórmula. Ella tiene que "aprender" la relación sola.

### Código

```python
from sklearn.linear_model import LinearRegression

# Ejemplos: números y su doble
X = [[1], [2], [3], [4], [5]]
y = [2, 4, 6, 8, 10]

# Creamos el modelo y lo entrenamos con los ejemplos
modelo = LinearRegression()
modelo.fit(X, y)

# Le preguntamos por un número que no vio
prediccion = modelo.predict([[10]])

print(prediccion)
```

### Salida

```
[20.]
```

### Explicación línea por línea

- `from sklearn.linear_model import LinearRegression`: importa un modelo de Machine Learning ya armado, que aprende relaciones lineales (del tipo "y = algo por x").
- `X = [[1], [2], [3], [4], [5]]`: los números de entrada. Van en una lista de listas porque el modelo espera poder recibir varias columnas de datos, aunque acá usemos una sola.
- `y = [2, 4, 6, 8, 10]`: el resultado esperado para cada número de `X` (el doble de cada uno).
- `modelo = LinearRegression()`: crea el modelo, todavía sin entrenar.
- `modelo.fit(X, y)`: acá ocurre el "aprendizaje". El modelo mira los ejemplos (`X` e `y`) y descubre solo la relación entre ellos.
- `modelo.predict([[10]])`: le preguntamos qué resultado daría para el número 10, un valor que nunca vio en los ejemplos.
- El modelo responde `20`, es decir, dedujo la regla "el doble" sin que se la escribiéramos en el código.

### Consigna

Cambiá `y` para que sea el triple de cada número (`[3, 6, 9, 12, 15]`), volvé a entrenar el modelo, y preguntale por el 10. ¿Qué responde?

---

## Cierre

Esto es la idea central de Machine Learning: en vez de escribir la fórmula a mano (`resultado = numero * 2`), le damos ejemplos al modelo y él encuentra la relación solo. En el resto del curso vamos a ver cómo se hace esto con datos reales, más variables, y problemas más complejos que "el doble de un número".
