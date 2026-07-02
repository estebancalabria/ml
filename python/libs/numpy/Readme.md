# NumPy: Guía didáctica

Repositorio de aprendizaje sobre **NumPy**, la librería fundamental para computación numérica en Python. Base de librerías como Pandas, Scikit-learn, TensorFlow y PyTorch.

## ¿Qué es NumPy?

NumPy (Numerical Python) provee un objeto de arreglo N-dimensional (`ndarray`) mucho más rápido y eficiente en memoria que las listas nativas de Python, gracias a que sus operaciones están implementadas en C.

## Instalación

```bash
pip install numpy
```

## Uso básico

```python
import numpy as np

# Crear un array
a = np.array([1, 2, 3, 4, 5])
print(a)

# Array 2D (matriz)
b = np.array([[1, 2, 3], [4, 5, 6]])
print(b.shape)  # (2, 3)
```

## Conceptos clave

### 1. Arrays vs listas de Python

```python
lista = [1, 2, 3]
array = np.array([1, 2, 3])

# Con listas hay que usar loops o comprensión
lista_x2 = [x * 2 for x in lista]

# Con arrays, la operación se aplica a todo el array (vectorización)
array_x2 = array * 2
```

### 2. Creación de arrays

```python
np.zeros((3, 3))        # matriz de ceros
np.ones((2, 4))         # matriz de unos
np.arange(0, 10, 2)     # [0, 2, 4, 6, 8]
np.linspace(0, 1, 5)    # 5 valores equiespaciados entre 0 y 1
np.random.rand(3, 3)    # matriz aleatoria 3x3
```

### 3. Indexing y slicing

```python
a = np.array([10, 20, 30, 40, 50])

a[0]        # 10
a[-1]       # 50
a[1:4]      # [20, 30, 40]
a[a > 20]   # filtrado booleano: [30, 40, 50]
```

### 4. Operaciones y broadcasting

```python
a = np.array([1, 2, 3])
b = np.array([10, 20, 30])

a + b        # [11, 22, 33]
a * 2        # [2, 4, 6] -> broadcasting
np.dot(a, b) # producto punto: 140
```

### 5. Funciones estadísticas

```python
a = np.array([1, 2, 3, 4, 5])

a.mean()   # promedio
a.std()    # desvío estándar
a.sum()    # suma
a.max()    # máximo
a.min()    # mínimo
```

### 6. Reshape

```python
a = np.arange(12)
a.reshape(3, 4)   # reorganiza en matriz 3x4
```

### 7. Álgebra lineal

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

np.matmul(A, B)      # multiplicación de matrices (o A @ B)
np.linalg.inv(A)     # matriz inversa
np.linalg.det(A)     # determinante
np.linalg.eig(A)     # autovalores y autovectores
```

### 8. Tipos de datos (dtype)

Cada array tiene un tipo de dato fijo, a diferencia de las listas de Python:

```python
a = np.array([1, 2, 3], dtype=np.int32)
b = np.array([1.5, 2.5], dtype=np.float64)

a.dtype   # dtype('int32')
a.astype(np.float64)  # conversión de tipo
```

Elegir el dtype correcto (int8, int32, float32, float64, etc.) impacta directamente en el uso de memoria, algo clave cuando se trabaja con datasets grandes.

### 9. Ejes (axis)

Muchas funciones de NumPy aceptan un parámetro `axis` para operar por filas o columnas en vez de sobre todo el array:

```python
m = np.array([[1, 2, 3], [4, 5, 6]])

m.sum()          # 21 (suma todo)
m.sum(axis=0)    # [5, 7, 9]  -> suma por columnas
m.sum(axis=1)    # [6, 15]    -> suma por filas
```

### 10. Vistas vs copias

Un punto que suele confundir: el slicing en NumPy devuelve una **vista**, no una copia. Modificar la vista modifica el array original.

```python
a = np.array([1, 2, 3, 4])
b = a[1:3]
b[0] = 99
print(a)  # [1, 99, 3, 4] -> a también cambió

c = a[1:3].copy()  # esto sí crea una copia independiente
```

## ¿Por qué usar NumPy en vez de listas?

- **Velocidad**: las operaciones están vectorizadas e implementadas en C, evitando el overhead de los loops interpretados de Python.
- **Memoria**: un array de NumPy ocupa un bloque contiguo de memoria con un tipo de dato fijo, mucho más compacto que una lista de objetos Python.
- **Funcionalidad**: álgebra lineal, generación de números aleatorios, transformadas de Fourier, estadística, manejo de arrays multidimensionales.
- **Base del ecosistema**: Pandas, Scikit-learn, Matplotlib, TensorFlow y PyTorch usan arrays de NumPy (o estructuras compatibles) internamente.

## Errores comunes al empezar

- Confundir `*` (multiplicación elemento a elemento) con `@` o `np.matmul` (multiplicación matricial).
- Olvidar que el slicing devuelve una vista y no una copia.
- Sumar/operar arrays de shapes incompatibles sin entender las reglas de broadcasting.
- Usar loops de Python sobre arrays en vez de operaciones vectorizadas (pierde toda la ventaja de rendimiento de NumPy).

## Recursos

- [Documentación oficial de NumPy](https://numpy.org/doc/stable/)
- [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html)

## Licencia

MIT
