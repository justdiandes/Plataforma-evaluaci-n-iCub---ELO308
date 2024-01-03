import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

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


while not rospy.is_shutdown():

    # Crear el objeto para capturar video

    results = model.track(frame, conf=0.5, device=0, persist=True)

    annotated_frame = results[0].plot()
    cv2.imshow("Imagen recibida", annotated_frame)



    if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    

rospy.signal_shutdown("Cerrando el nodo")
