import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np
import cvzone
from cvzone.ColorModule import ColorFinder
import math
from std_msgs.msg import Float64MultiArray
import yarp
import numpy
rospy.init_node('icub_node', anonymous=True)

pub = rospy.Publisher('/datos_sensor2', Float64MultiArray, queue_size=10)

yarp.Network.init()



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



# prepare a property object
props = yarp.Property()
props.put("device","remote_controlboard")
props.put("local","/client/head")
props.put("remote","/icubSim/head")

headDriver = yarp.PolyDriver(props)
#query motor control interfaces
iPos = headDriver.viewIPositionControl()
iEnc = headDriver.viewIEncoders()

#retrieve number of joints
jnts=iPos.getAxes()

#retrieve number of joints
jnts=iPos.getAxes()
 
print('Controlling', jnts, 'joints')
 
# read encoders
encs=yarp.Vector(jnts)
iEnc.getEncoders(encs.data())

# store as home position
home=yarp.Vector(jnts, encs.data())
 
#initialize a new tmp vector identical to encs
tmp=yarp.Vector(jnts)
tmp.set(0, tmp.get(0)-40)
iPos.positionMove(tmp.data()) 

'''
output_port = yarp.Port()
output_port.open("/python-image-port")
yarp.Network.connect("/python-image-port", "/view01")
output_port.write(yarp_image)
'''
# Create a port and connect it to the iCub simulator virtual camera
input_port = yarp.Port()
input_port.open("/python-image-port")		

yarp.Network.connect("/icubSim/cam/right/rgbImage:o", "/cam1")
yarp.Network.connect("/icubSim/cam/right/rgbImage:o", "/python-image-port")

img_array = numpy.zeros((240, 320, 3), dtype=numpy.uint8)
yarp_image = yarp.ImageRgb()
yarp_image.resize(320, 240)
yarp_image.setExternal(img_array, img_array.shape[1], img_array.shape[0])
input_port.read(yarp_image)

#------------------Conectividad con ROS y Gazebo-------------------------#
ref_image = cv2.imread("conocida_805.png")
#--------------Matríz de parámetros intrínsecos-------------------#
focal_length_x = 215.483
focal_length_y = 214.935
optical_center_x = 174.868
optical_center_y = 105.63
K = np.array([[focal_length_x, 0, optical_center_x],
              [0, focal_length_y, optical_center_y],
              [0, 0, 1]])


theta = (50)/(math.pi*180)

Rot_m = np.array([[math.cos(theta), 0, math.sin(theta)],
                  [0, 1, 0],
                  [-math.sin(theta), 0, math.cos(theta)]])

trans_v = np.array([0.693799, 0.000005, -0.338206])

T = np.array([[math.cos(theta), 0, math.sin(theta), 0.693799],
              [0, 1, 0, 0.000005],
              [-math.sin(theta), 0, math.cos(theta), -0.338206],
              [0, 0, 0, 1]])

T_inverse = np.linalg.inv(T)

colorF = ColorFinder(False)

hsvVals_sim = {'hmin': 0, 'smin': 34, 'vmin': 0, 'hmax': 4, 'smax': 255, 'vmax': 255}


#--------------Extracción datos desde imagen referencia para cálculo de distancia------------------#






while not rospy.is_shutdown():
    yarp_image = yarp.ImageRgb()
    yarp_image.resize(320, 240)
    yarp_image.setExternal(img_array, img_array.shape[1], img_array.shape[0])
    input_port.read(yarp_image)
    frame = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)


    # Crear el objeto para capturar video

    # Crear el objeto para capturar video
    h, w,_ = frame.shape

    imgColor, mask = colorF.update(frame, hsvVals_sim)
    #--------Función cvzone permite obtener los bounding boxes automáticamente------#
    imgContour, contours = cvzone.findContours(frame, mask, minArea=100) #Utiliza cv2 para ser creada pero cvzone automatiza este proceso

    
    if contours: #aquí se obtiene la coordenada del punto central del bounding box en (x,y) y se obtiene el parámetro de área del bounding box
        data = contours[0]['center'][0],\
                h - contours[0]['center'][1], \
                int(contours[0]['area']) #Lista de contornos -> Queremos el controrno más grande
        #print(data)

    z = round((focal_length_x * 0.07) / (diametro_pixel(data[2])), 2)
    y = round(((data[0]-optical_center_x)*z)/focal_length_x, 2)
    x = round(((data[1]-optical_center_y)*z)/focal_length_y, 2)
    #both2 = np.concatenate((mask, imgContour), axis=1)
    object_coords = (x, y, z)
    coords_hom_robot = np.array([x, y, z])
    coords_robot_cam_rot = np.dot(Rot_m, coords_hom_robot)
    coords_robot_cam = coords_robot_cam_rot + trans_v
    
    #coords_robot_cam = np.dot(T_inverse, coords_hom_robot)
    #coords_robot_cam = np.dot(coords_hom_robot, T)
    
    x_final = round(coords_robot_cam[0], 2)
    y_final = round(coords_robot_cam[1], 2)
    z_final = round(coords_robot_cam[2], 2)
    datos = np.array([z_final, y_final, x_final])
    escribir_a_ros(datos)
  
    cv2.putText(imgContour, f"Coordenadas: ({y_final}, {z_final}, {x_final})", (50,50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 1)
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
