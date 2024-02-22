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

data = 'validacion_detYOLO2.csv'
print(data)
df = pd.read_csv(data, delimiter=',', header=None)
for index, row in df.iterrows():
    label = row[1]
    x, y, z = (float(row[2])), (float(row[3])), (float(row[4]))

    if label == 'cam':
        if last_label != 'cam':
            cam_i += 1
            cam_df = pd.concat([cam_df, pd.DataFrame({'X': [x], 'Y': [y], 'Z': [z]})], ignore_index=True)
    elif label == 'real':
        if last_label != 'real':
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

# Calcula la varianza y desviación estándar de las diferencias en cada dimensión
varianza_diferencias = diferencias.var()
desviacion_estandar_diferencias = diferencias.std()

# Imprime los resultados
print("Varianza de las diferencias en cada dimensión (x, y, z):\n", varianza_diferencias)
print("\nDesviación estándar de las diferencias en cada dimensión (x, y, z):\n", desviacion_estandar_diferencias)

mean_x = np.mean((cam_x - icub_x)**2)
mean_y = np.mean((cam_y - icub_y)**2)
mean_z = np.mean((cam_z - icub_z)**2)

print(f"error: X: {mean_x}, Y: {mean_y}, Z: {mean_z}")


print(icub_df)

mean = np.mean((cam_x - icub_x) ** 2 + (cam_y - icub_y) ** 2 + (cam_z - icub_z) ** 2)
#mean = np.mean(np.abs(cam_x - icub_x) + np.abs(cam_y - icub_y) + np.abs(cam_z - icub_z))
rmean = np.sqrt(mean)

print(f"error: {mean}")

#Plot icub and cam points
ax.scatter(icub_df['X'], icub_df['Y'], icub_df['Z'], c='b', marker='o', label='real')
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
ax.plot(icub_df.index, icub_x, c='b', marker='o', label='real')
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
ax.plot(icub_df.index, icub_y, c='b', marker='o', label='real')
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
ax.plot(icub_df.index, icub_z, c='b', marker='o', label='real')
ax.plot(cam_df.index, cam_z, c='r', marker='o', label='cam')

ax.set_title('Eje z')

ax.set_xlabel('frames')
ax.set_ylabel('metros')

ax.set_xlim([-1, 55])

ax.legend()
plt.show()