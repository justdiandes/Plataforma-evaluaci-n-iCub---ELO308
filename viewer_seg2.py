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
from std_msgs.msg import Float64MultiArray




pub = rospy.Publisher('/datos_sensor1', Float64MultiArray, queue_size=10)


def escribir_a_ros(datos):
    # Publicar datos en el tópico de ROS
    pub.publish(Float64MultiArray(data=datos))


def diametro_pixel(area):
    radio = math.sqrt(area / math.pi)
    diametro = 2 * radio
    return diametro


KNOWN_DISTANCE = 0.805 #centimeters
KNOWN_WIDTH = 0.07 #centimeters

def Focal_length(measured_distance, real_width, width_in_rf_image):
    focal_length = (width_in_rf_image * measured_distance)/real_width
    return focal_length

def Distance_finder(Focal_Length, real_object_width, object_width_in_frame):    
    distance = (real_object_width * Focal_Length)/object_width_in_frame
    return distance


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
ref_image = cv2.imread("conocida_80.png")
#--------------Matríz de parámetros intrínsecos-------------------#
focal_length_x = 1663.1481384679323
focal_length_y = 1663.1481384679323
optical_center_x = 960.5
optical_center_y = 540.5
K = np.array([[focal_length_x, 0, optical_center_x],
              [0, focal_length_y, optical_center_y],
              [0, 0, 1]])

colorF = ColorFinder(False)

hsvVals_sim = {'hmin': 0, 'smin': 34, 'vmin': 0, 'hmax': 4, 'smax': 255, 'vmax': 255}


#--------------Extracción datos desde imagen referencia para cálculo de distancia------------------#

imgColoRef, maskRef = colorF.update(ref_image, hsvVals_sim)
imgContourRef, contoursRef = cvzone.findContours(ref_image, maskRef, minArea=2000)
href, wref, _ = ref_image.shape
if contoursRef:
    data = contoursRef[0]['center'][0],\
            href - contoursRef[0]['center'][1],\
            int(contoursRef[0]['area'])

object_width_image = diametro_pixel(data[2])

cv2.imshow("Ref", imgContourRef)

Focal_length_found = Focal_length(KNOWN_DISTANCE, KNOWN_WIDTH, object_width_image)


#Podemos estimar el focal length o usar el focal length x (el valor no varía tanto c:)

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

    z = round((Focal_length_found* 0.07) / (diametro_pixel(data[2])), 3)
    y = round(((data[0]-optical_center_x)*z)/focal_length_x, 3)
    x = round(((data[1]-optical_center_y)*z)/focal_length_y, 3)
    #both2 = np.concatenate((mask, imgContour), axis=1)
    object_coords = (-x, y, z)
    datos = np.array([object_coords[0], object_coords[1], 1.31 - object_coords[2]])
    escribir_a_ros(datos)
  
    cv2.putText(imgContour, f"Coordenadas: ({object_coords[0]}, {object_coords[1]}, {object_coords[2]})", (50,50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 1)
    #cv2.putText(imgContour, f"Distancia [cm] = {Distance}",(50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    #cv2.putText(imgContour, f"Distancia [cm] = {z_coordinate}",(50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    both1 = np.concatenate((frame, imgColor), axis=1)

    print(f"({object_coords[0]}, {object_coords[1]},{object_coords[2]})")

    cv2.imshow("Image color", both1)
    cv2.imshow("Color detection", imgContour)



    if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    

rospy.signal_shutdown("Cerrando el nodo")
