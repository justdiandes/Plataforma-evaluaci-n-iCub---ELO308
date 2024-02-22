import csv
import numpy as np

def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)

def frechet_distance(P, Q):
    n = len(P)
    m = len(Q)
    
    if n == 0 or m == 0:
        return float('inf')  # Si una de las trayectorias está vacía, la distancia de Fréchet es infinita
    
    # Matriz de distancia
    D = np.zeros((n, m))
    
    # Calcular la distancia entre cada par de puntos
    for i in range(n):
        for j in range(m):
            D[i][j] = distance(P[i], Q[j])
    
    # Calcular la distancia de Fréchet utilizando programación dinámica
    M = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                M[i][j] = D[0][0]
            elif i == 0:
                M[i][j] = max(M[0][j-1], D[0][j])
            elif j == 0:
                M[i][j] = max(M[i-1][0], D[i][0])
            else:
                M[i][j] = max(min(M[i-1][j], M[i-1][j-1], M[i][j-1]), D[i][j])
    
    return M[n-1][m-1]

icub_points = []
cam_points = []

with open('icub_detYOLO_sin_cambio_z.csv') as csvfile:
    csvReader = csv.reader(csvfile, delimiter=',')
    for row in csvReader:
        label = row[1]
        x, y, z = float(row[2]), float(row[3]), float(row[4])

        if label == 'cam':
            cam_points.append((x, y, z))
        elif label == 'icub':
            icub_points.append((x, y, z))

# Calcular la distancia de Fréchet entre las trayectorias icub y cam
frechet_dist = frechet_distance(icub_points, cam_points)
print("Distancia de Fréchet:", frechet_dist)
