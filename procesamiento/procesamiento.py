import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

icub_i = 0
cam_i = 0

# DataFrames para almacenar puntos 3D
icub_df = pd.DataFrame(columns=['X', 'Y', 'Z'])
cam_df = pd.DataFrame(columns=['X', 'Y', 'Z'])

last_label = None  # Para verificar el último label leído

data = 'procesamiento/icub_segColor.csv'
print(data)
df = pd.read_csv(data, delimiter=',', header=None)
for index, row in df.iterrows():
    label = row[1]
    x, y, z = (float(row[2])), (float(row[3])), (float(row[4]))

    if label == 'cam':
        if last_label != 'cam':
            cam_i += 1
            cam_df = pd.concat([cam_df, pd.DataFrame({'X': [x], 'Y': [y], 'Z': [z]})], ignore_index=True)
    elif label == 'icub':
        if last_label != 'icub':
            icub_i += 1
            icub_df = pd.concat([icub_df, pd.DataFrame({'X': [x], 'Y': [y], 'Z': [z]})], ignore_index=True)

    last_label = label

# Limitar la longitud de puntos al mínimo existente
min_length = min(len(icub_df), len(cam_df))
icub_df = icub_df.head(2000)
cam_df = cam_df.head(2000)

icub_x = icub_df['X'].values
cam_x = cam_df['X'].values

icub_y = icub_df['Y'].values
cam_y = cam_df['Y'].values

icub_z = icub_df['Z'].values
cam_z = cam_df['Z'].values

# Calcula las diferencias entre los datos experimentales y teóricos en cada dimensión
diferencias = cam_df - icub_df


mean_x = np.mean((cam_x - icub_x)**2)
mean_y = np.mean((cam_y - icub_y)**2)
mean_z = np.mean((cam_z - icub_z)**2)

print(f"error: X: {mean_x}, Y: {mean_y}, Z: {mean_z}")


print(cam_df)

mean = np.mean((cam_x - icub_x) ** 2 + (cam_y - icub_y) ** 2 + (cam_z - icub_z) ** 2)
#mean = np.mean(np.abs(cam_x - icub_x) + np.abs(cam_y - icub_y) + np.abs(cam_z - icub_z))
rmean = np.sqrt(mean)

print(f"error: {mean}")

#Plot icub and cam points
ax.scatter(icub_df['X'], icub_df['Y'], icub_df['Z'], c='b', marker='o', label='icub')
ax.scatter(cam_df['X'], cam_df['Y'], cam_df['Z'], c='r', marker='o', label='cam')

ax.set_xlabel('Eje X')
ax.set_ylabel('Eje Y')
ax.set_zlabel('Eje Z')

ax.set_xlim([-1, 0.5])
ax.set_ylim([-1, 0.5])
ax.set_zlim([0, 1])

ax.legend()
plt.show()

print("Conteo de datos para icub:", len(icub_df['Z']))
print("Conteo de datos para cam:", len(cam_df['Z']))

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)

# Plot icub and cam points on the X axis only
ax.plot(icub_df.index, icub_x, c='b', marker='o', label='icub')
ax.plot(cam_df.index, cam_x, c='r', marker='o', label='cam')

ax.set_title('Eje x')

ax.set_xlabel('frames')
ax.set_ylabel('metros')

ax.set_xlim([-1, 55])

ax.legend()
plt.show()

print("Conteo de datos para icub:", len(icub_df['Z']))
print("Conteo de datos para cam:", len(cam_df['Z']))

fig = plt.figure()
ax = fig.add_subplot(111)
# Plot icub and cam points on the X axis only
ax.plot(icub_df.index, icub_y, c='b', marker='o', label='icub')
ax.plot(cam_df.index, cam_y, c='r', marker='o', label='cam')

ax.set_title('Eje y')

ax.set_xlabel('frames')
ax.set_ylabel('metros')

ax.set_xlim([-1, 55])

ax.legend()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
# Plot icub and cam points on the X axis only
ax.plot(icub_df.index, icub_z, c='b', marker='o', label='icub')
ax.plot(cam_df.index, cam_z, c='r', marker='o', label='cam')

ax.set_title('Eje z')

ax.set_xlabel('frames')
ax.set_ylabel('metros')

ax.set_xlim([-1, 55])

ax.legend()
plt.show()


error_porcentual_arr_x = np.abs(icub_x - cam_x)*100
error_porcentual_arr_y = np.abs(icub_y - cam_y)*100
error_porcentual_arr_z = np.abs(icub_z - cam_z)*100


error_porcentual_total_x = np.mean(error_porcentual_arr_x)
error_porcentual_total_y = np.mean(error_porcentual_arr_y)
error_porcentual_total_z = np.mean(error_porcentual_arr_z)

err_total = (error_porcentual_total_x + error_porcentual_total_y + error_porcentual_total_z)/3

# Imprime los resultados
print("Porcentaje de error en la dimensión X:", error_porcentual_total_x, "%")
print("Porcentaje de error en la dimensión Y:", error_porcentual_total_y, "%")
print("Porcentaje de error en la dimensión Z:", error_porcentual_total_z, "%")
print("Porcentaje de error en la dimensión Z:", err_total, "%")


residuos = icub_x - cam_x

# Supongamos que 'residuos' es una lista o arreglo que contiene los residuos calculados
# a partir de los valores observados y predichos por el modelo.

# Crear el histograma de residuos
plt.hist(residuos, bins=10, edgecolor='black')  # Puedes ajustar el número de bins según tus preferencias
plt.xlabel('Residuos')
plt.ylabel('Frecuencia')
plt.title('Histograma de Residuos eje x')
plt.grid(True)
plt.show()


residuos = icub_y - cam_y

# Supongamos que 'residuos' es una lista o arreglo que contiene los residuos calculados
# a partir de los valores observados y predichos por el modelo.

# Crear el histograma de residuos
plt.hist(residuos, bins=10, edgecolor='black')  # Puedes ajustar el número de bins según tus preferencias
plt.xlabel('Residuos')
plt.ylabel('Frecuencia')
plt.title('Histograma de Residuos eje y')
plt.grid(True)
plt.show()


residuos = icub_z - cam_z

# Supongamos que 'residuos' es una lista o arreglo que contiene los residuos calculados
# a partir de los valores observados y predichos por el modelo.

# Crear el histograma de residuos
plt.hist(residuos, bins=10, edgecolor='black')  # Puedes ajustar el número de bins según tus preferencias
plt.xlabel('Residuos')
plt.ylabel('Frecuencia')
plt.title('Histograma de Residuos eje z')
plt.grid(True)
plt.show()




x = cam_x  # Primeros cinco elementos en el eje x
y = icub_x  # Primeros cinco elementos en el eje y


# Graficar los puntos

plt.plot(x, y, 'o', label='Datos')
plt.xlabel('Valor Real')
plt.ylabel('Valor Obtenido')
plt.title('Diagrama')
plt.grid(True)

# Agregar la recta y = x
x = np.linspace(min(x), max(x), 100)  # 100 puntos entre 0 y el máximo valor esperado
plt.plot(x, x, '--', label='y = x', color='red')  # Dibujar la recta y = x en rojo, con línea punteada

# Mostrar leyenda
plt.legend()

plt.show()


