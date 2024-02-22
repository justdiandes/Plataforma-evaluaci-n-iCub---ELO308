import yarp
import numpy
import cv2 as cv
from ultralytics import YOLO
import numpy as np
import math
from std_msgs.msg import Float64MultiArray
import rospy


model = YOLO("/home/diego/Escritorio/Plataforma-evaluacion-iCub-ELO308/runs/segment/train/weights/last.pt")
rospy.init_node('icub_node', anonymous=True)


pub = rospy.Publisher('/datos_sensor2', Float64MultiArray, queue_size=10)

yarp.Network.init()



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

ref_image = cv.imread("conocida_80.png")
result = model.predict(ref_image, device=0)


mask_raw = result[0].masks[0].cpu().data.numpy().transpose(1, 2, 0)
    
    # Convert single channel grayscale to 3 channel image
mask_3channel = cv.merge((mask_raw,mask_raw,mask_raw))

    # Get the size of the original image (height, width, channels)
h2, w2, c2 = result[0].orig_img.shape
    
    # Resize the mask to the same size as the image (can probably be removed if image is the same size as the model)
mask = cv.resize(mask_3channel, (w2, h2))

    # Convert BGR to HSV
hsv = cv.cvtColor(mask, cv.COLOR_BGR2HSV)

    # Define range of brightness in HSV
lower_black = np.array([0,0,0])
upper_black = np.array([0,0,1])
  # Create a mask. Threshold the HSV image to get everything black
mask = cv.inRange(mask, lower_black, upper_black)

    # Invert the mask to get everything but black
mask = cv.bitwise_not(mask)

cv.imshow("ref", mask)
contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Inicializar el área total
total_area = 0

# Iterar sobre los contornos y calcular el área de cada uno
for contour in contours:
    area = cv.contourArea(contour)
    total_area += area

object_width_image = diametro_pixel(total_area)


Focal_length_found = Focal_length(KNOWN_DISTANCE, KNOWN_WIDTH, object_width_image)

print(Focal_length_found)


print(f"area: {total_area}, ancho: {object_width_image}")

while(1):
    yarp_image = yarp.ImageRgb()
    yarp_image.resize(320, 240)
    yarp_image.setExternal(img_array, img_array.shape[1], img_array.shape[0])
    input_port.read(yarp_image)
    frame = cv.cvtColor(img_array, cv.COLOR_BGR2RGB)

    #----------YOLO implementation-----------#

    results = model.predict(frame, device=0)

    #annotated_frame = results[0].plot()

    #cv.imshow("mask", (results[0].masks[0].cpu().data.numpy().transpose(1, 2, 0) * 255).astype("uint8"))

    if(results[0].masks is not None):
        # Convert mask to single channel image
        mask_raw = results[0].masks[0].cpu().data.numpy().transpose(1, 2, 0)
        
        # Convert single channel grayscale to 3 channel image
        mask_3channel = cv.merge((mask_raw,mask_raw,mask_raw))

        # Get the size of the original image (height, width, channels)
        h2, w2, c2 = results[0].orig_img.shape
        
        # Resize the mask to the same size as the image (can probably be removed if image is the same size as the model)
        mask = cv.resize(mask_3channel, (w2, h2))

        # Convert BGR to HSV
        hsv = cv.cvtColor(mask, cv.COLOR_BGR2HSV)

        # Define range of brightness in HSV
        lower_black = np.array([0,0,0])
        upper_black = np.array([0,0,1])

        # Create a mask. Threshold the HSV image to get everything black
        mask = cv.inRange(mask, lower_black, upper_black)

        # Invert the mask to get everything but black
        mask = cv.bitwise_not(mask)
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Apply the mask to the original image
    total_area2 = 0

    # Iterar sobre los contornos y calcular el área de cada uno
    for contour in contours:
        area = cv.contourArea(contour)
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
            cv.circle(annotated_frame, center_point, 5, (0, 255, 255), -1)  # Punto en el centro en rojo
    w = x2 - x1
    h = y2 - y1
    z = (focal_length_x * 0.07) / w
    y = ((center_x-optical_center_x)*z)/focal_length_x
    x = ((center_y-optical_center_y)*z)/focal_length_y
    
    object_coords = (x, y, z)
    #coords_hom_robot = np.array([x, y, z, 1])
    coords_hom_robot = np.array([x, y, z])
    coords_robot_cam_rot = np.dot(Rot_m, coords_hom_robot)
    coords_robot_cam = coords_robot_cam_rot + trans_v
    
    #coords_robot_cam = np.dot(T_inverse, coords_hom_robot)
    #coords_robot_cam = np.dot(coords_hom_robot, T)
    
    x_final = round(coords_robot_cam[0], 2)
    y_final = round(coords_robot_cam[1], 2)
    z_final = round(coords_robot_cam[2], 2)
    datos = np.array([z_final, y_final, x_final])
    pub.publish(Float64MultiArray(data=datos))
    
    cv.putText(annotated_frame, f"Coordenadas: ({y_final}, {z_final}, {x_final})", (50,50),cv.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 1)
    cv.imshow("Imagen recibida", annotated_frame)
    print(f"({z_final}, {y_final}, {x_final})")



    if cv.waitKey(1) & 0xFF == ord('q'):
        break


cv.destroyAllWindows()
iPos.positionMove(home.data())