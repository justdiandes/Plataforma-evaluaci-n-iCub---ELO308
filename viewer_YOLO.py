import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np
import cvzone
import math

def calcular_diametro_desde_area(area):
    radio = math.sqrt(area / math.pi)
    diametro = 2 * radio
    return diametro


#---------------------------Importar YOLO--------------------------------#
model = YOLO('/home/diego/Escritorio/camera_ROS/train9_pc_icub/weights/best.pt')
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
ref_image = cv2.imread("conocida_805.png")
#--------------Matríz de parámetros intrínsecos-------------------#
focal_length_x = 1663.1481384679323
focal_length_y = 1663.1481384679323
optical_center_x = 960.5
optical_center_y = 540.5
K = np.array([[focal_length_x, 0, optical_center_x],
              [0, focal_length_y, optical_center_y],
              [0, 0, 1]])






while not rospy.is_shutdown():

    # Crear el objeto para capturar video

    results = model.track(frame, conf=0.5, device=0, persist=True)
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
    z = (focal_length_x * 0.07) / (calcular_diametro_desde_area(w*h))
    y = ((center_x-optical_center_x)*z)/focal_length_x
    x = ((center_y-optical_center_y)*z)/focal_length_y
    
    object_coords = (-x, -y, z)

    cv2.putText(annotated_frame, f"Coordenadas: ({object_coords[0]}, {object_coords[1]}, {object_coords[2]})", (50,50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 1)
    cv2.imshow("Imagen recibida", annotated_frame)
    print(f"({object_coords[0]}, {object_coords[1]}, {object_coords[2]})")

    if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    

rospy.signal_shutdown("Cerrando el nodo")
