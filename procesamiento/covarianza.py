import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar los datos desde el archivo CSV


data = 'validacion_segColor1.csv'

df = pd.read_csv(data, names=['Tiempo', 'Nombre', 'X', 'Y', 'Z'])
print(data)
# Filtrar los datos para obtener solo los de la cámara (cam) y los datos reales (real)
df_cam = df[df['Nombre'] == 'cam']
df_real = df[df['Nombre'] == 'icub']

# Determinar el mínimo de datos entre 'cam' y 'real'
min_length = min(len(df_cam), len(df_real))

# Limitar el largo de los datos al mínimo común
df_cam = df_cam.head(min_length)
df_real = df_real.head(min_length)

# Calcular las covarianzas para cada eje de coordenadas
covariance_x = np.trace(np.cov(df_cam['X'], df_real['X']))
covariance_y = np.trace(np.cov(df_cam['Y'], df_real['Y']))
covariance_z = np.trace(np.cov(df_cam['Z'], df_real['Z']))

print("Covarianza X:", covariance_x)
print("Covarianza Y:", covariance_y)
print("Covarianza Z:", covariance_z)

# Visualización gráfica de las magnitudes de las covarianzas
covariances = [covariance_x, covariance_y, covariance_z]
coordinates = ['X', 'Y', 'Z']

plt.bar(coordinates, covariances)
plt.title('Magnitud de la covarianza para cada eje de coordenadas')
plt.xlabel('Eje de coordenadas')
plt.ylabel('Magnitud de la covarianza')
plt.show()
