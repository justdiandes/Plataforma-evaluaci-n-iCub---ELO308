import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np
import cvzone
import math
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


#---------------------------Importar YOLO--------------------------------#
model = YOLO('/home/diego/Escritorio/Plataforma-evaluacion-iCub-ELO308/runs/segment/train/weights/last.pt')
frame = None
#---------------------------Importar YOLO--------------------------------#

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

result = model.predict(ref_image)



mask_raw = result[0].masks[0].cpu().data.numpy().transpose(1, 2, 0)
    
    # Convert single channel grayscale to 3 channel image
mask_3channel = cv2.merge((mask_raw,mask_raw,mask_raw))

    # Get the size of the original image (height, width, channels)
h2, w2, c2 = result[0].orig_img.shape
    
    # Resize the mask to the same size as the image (can probably be removed if image is the same size as the model)
mask = cv2.resize(mask_3channel, (w2, h2))

    # Convert BGR to HSV
hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)

    # Define range of brightness in HSV
lower_black = np.array([0,0,0])
upper_black = np.array([0,0,1])
  # Create a mask. Threshold the HSV image to get everything black
mask = cv2.inRange(mask, lower_black, upper_black)

    # Invert the mask to get everything but black
mask = cv2.bitwise_not(mask)

cv2.imshow("ref", mask)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Inicializar el área total
total_area = 0

# Iterar sobre los contornos y calcular el área de cada uno
for contour in contours:
    area = cv2.contourArea(contour)
    total_area += area

object_width_image = diametro_pixel(total_area)


Focal_length_found = Focal_length(KNOWN_DISTANCE, KNOWN_WIDTH, object_width_image)

print(Focal_length_found)


print(f"area: {total_area}, ancho: {object_width_image}")

while not rospy.is_shutdown():

    # Crear el objeto para capturar video

    results = model.predict(frame, conf=0.5, device=0)

    #annotated_frame = results[0].plot()

    #cv2.imshow("mask", (results[0].masks[0].cpu().data.numpy().transpose(1, 2, 0) * 255).astype("uint8"))

    if(results[0].masks is not None):
        # Convert mask to single channel image
        mask_raw = results[0].masks[0].cpu().data.numpy().transpose(1, 2, 0)
        
        # Convert single channel grayscale to 3 channel image
        mask_3channel = cv2.merge((mask_raw,mask_raw,mask_raw))

        # Get the size of the original image (height, width, channels)
        h2, w2, c2 = results[0].orig_img.shape
        
        # Resize the mask to the same size as the image (can probably be removed if image is the same size as the model)
        mask = cv2.resize(mask_3channel, (w2, h2))

        # Convert BGR to HSV
        hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)

        # Define range of brightness in HSV
        lower_black = np.array([0,0,0])
        upper_black = np.array([0,0,1])

        # Create a mask. Threshold the HSV image to get everything black
        mask = cv2.inRange(mask, lower_black, upper_black)

        # Invert the mask to get everything but black
        mask = cv2.bitwise_not(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Apply the mask to the original image
    total_area2 = 0

    # Iterar sobre los contornos y calcular el área de cada uno
    for contour in contours:
        area = cv2.contourArea(contour)
        total_area2 += area

    
    annotated_frame = results[0].plot()
    for result in results:
        boxes = result.boxes.cpu().numpy() # get boxes on cpu in numpy
        for box in boxes: # iterate boxes
            r = box.xyxy[0].astype(int) # get corner points as int
            x1 = int(r[0])
            y1 = int(r[1])
            x2 = int(r[2])
            y2 = int(r[3])
            # Calcular el centro del bounding box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            center_point = (center_x, center_y)
            cv2.circle(annotated_frame, center_point, 5, (0, 255, 255), -1)  # Punto en el centro en rojo
    w = x2 - x1
    h = y2 - y1

    z = round((Focal_length_found * 0.07) / (diametro_pixel(total_area2)), 3)
    y = round(((center_x-optical_center_x)*z)/focal_length_x, 3)
    x = round(((center_y-optical_center_y)*z)/focal_length_y, 3)

    object_coords = (x, y, z)
    datos = np.array([object_coords[0], object_coords[1], 1.31 - object_coords[2]])
    escribir_a_ros(datos)
  
    cv2.putText(annotated_frame, f"Coordenadas: ({object_coords[0]}, {object_coords[1]}, {object_coords[2]})", (50,50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 1)
    # Show the masked part of the image
    cv2.imshow("Imagen recibida", annotated_frame)



    if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    

rospy.signal_shutdown("Cerrando el nodo")
