import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import cvzone
import math
from cvzone.ColorModule import ColorFinder
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

KNOWN_DISTANCE = 0.805 #centimeters
KNOWN_WIDTH = 0.07 #centimeters

def Focal_length(measured_distance, real_width, width_in_rf_image):
    focal_length = (width_in_rf_image * measured_distance)/real_width
    return focal_length

def Distance_finder(Focal_Length, real_object_width, object_width_in_frame):    
    distance = (real_object_width * Focal_Length)/object_width_in_frame
    return distance

def ancho_pixel(area):
    radio = math.sqrt(area / math.pi)
    diametro = 2 * radio
    return diametro
#------------------Conectividad con ROS y Gazebo-------------------------#

# Definir la función image_callback
def image_callback(image_msg):
    global frame
    # Convertir el mensaje de imagen a una imagen de OpenCV
    bridge = CvBridge()
    frame = bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")

    # Ahora puedes trabajar con la imagen de OpenCV (cv_image)
    # por ejemplo, mostrarla

# Inicializar el nodo de ROS
rospy.init_node('camera_subscriber', anonymous=True)

# Crear un objeto CvBridge
bridge = CvBridge()

# Suscribirse al tópico de la cámara de ROS
image_sub = rospy.Subscriber('/my_camera/image_raw', Image, image_callback)

#------------------Conectividad con ROS y Gazebo-------------------------#

#------------------Segmentación inicial-------------------------#

ref_image = cv2.imread("conocida_805.png")

colorF = ColorFinder(False)
#Hue, saturation, value
#hsvVals = {'hmin': 0, 'smin': 91, 'vmin': 124, 'hmax': 6, 'smax': 255, 'vmax': 255}# Este es el primero, funciona con menos luminosidad
#Buscar la forma de tener un rango de detección
#hsvVals = {'hmin': 0, 'smin': 74, 'vmin': 117, 'hmax': 179, 'smax': 255, 'vmax': 255}
hsvVals_sim = {'hmin': 0, 'smin': 34, 'vmin': 0, 'hmax': 4, 'smax': 255, 'vmax': 255}


#--------------Matríz de parámetros intrínsecos-------------------#
focal_length_x = 1663.1481384679323
focal_length_y = 1663.1481384679323
optical_center_x = 960.5
optical_center_y = 540.5
K = np.array([[focal_length_x, 0, optical_center_x],
              [0, focal_length_y, optical_center_y],
              [0, 0, 1]])

#--------------Extracción datos desde imagen referencia para cálculo de distancia------------------#

imgColoRef, maskRef = colorF.update(ref_image, hsvVals_sim)
imgContourRef, contoursRef = cvzone.findContours(ref_image, maskRef, minArea=2000)
href, wref, _ = ref_image.shape
if contoursRef:
    data = contoursRef[0]['center'][0],\
            href - contoursRef[0]['center'][1],\
            int(contoursRef[0]['area'])

object_width_image = data[2]
'''
else:
    # Asignar un valor predeterminado o manejar el caso en el que no se encuentren contornos
    object_width_image = 0  # o cualquier otro valor predeterminado que tenga sentido en tu aplicación
    data = (0, 0, 0)  # Otra opción para manejar el caso sin contornos
'''
cv2.imshow("Ref", imgContourRef)


Focal_length_found = Focal_length(KNOWN_DISTANCE, KNOWN_WIDTH, object_width_image)

Base_distance = 80 #cm


#------------------Segmentación inicial-------------------------#

while not rospy.is_shutdown():

    # Crear el objeto para capturar video
    h, w,_ = frame.shape

    imgColor, mask = colorF.update(frame, hsvVals_sim)
    #--------Función cvzone permite obtener los bounding boxes automáticamente------#
    imgContour, contours = cvzone.findContours(frame, mask, minArea=5000) #Utiliza cv2 para ser creada pero cvzone automatiza este proceso

    if contours: #aquí se obtiene la coordenada del punto central del bounding box en (x,y) y se obtiene el parámetro de área del bounding box
        data = contours[0]['center'][0],\
                h - contours[0]['center'][1], \
                int(contours[0]['area']) #Lista de contornos -> Queremos el controrno más grande
        #print(data)

    
    #both2 = np.concatenate((mask, imgContour), axis=1)
    if data[2] != 0:
        Distance = Distance_finder(Focal_length_found, KNOWN_WIDTH, ancho_pixel(data[2]))
    else:
        Distance = 0
    #--------------------------Cálculo de coordenadas--------------------------------#
    z_coordinate = Distance
    u_frame = data[0]
    v_frame = data[1]
    
    #proyectar puntos 2D en plano imagen
    #image_plane_coords = np.array([[u_frame], [v_frame], [1]])
    #Normalizar coordenadas 2D
    #normalized_coords = np.dot(np.linalg.inv(K), image_plane_coords)
    #Cálculo coordenadas 3D
    #object_coords = image_plane_coords
    #object_coords = normalized_coords * z_coordinate

    x = ((u_frame - optical_center_x)/(focal_length_x))*z_coordinate
    y = ((v_frame - optical_center_y)/(focal_length_y))*z_coordinate
    object_coords = (x, y, z_coordinate)

    #object_coords = object_coords.round(5)
    #object_coords = normalized_coords
    print(f"({object_coords[0]}, {object_coords[1]}, {object_coords[2]})")

    cv2.putText(imgContour, f"Coordenadas: ({object_coords[0]}, {object_coords[1]}, {object_coords[2]})", (50,50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 1)
    cv2.putText(imgContour, f"Distancia [cm] = {Distance}",(50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(imgContour, f"Distancia [cm] = {z_coordinate}",(50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    both1 = np.concatenate((frame, imgColor), axis=1)



    cv2.imshow("Image color", both1)
    cv2.imshow("Color detection", imgContour)



    if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    

rospy.signal_shutdown("Cerrando el nodo")
