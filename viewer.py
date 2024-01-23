import yarp
import numpy
import cv2 as cv
from ultralytics import YOLO
import numpy as np
import math
import csv

model = YOLO("/home/diego/Escritorio/Plataforma-evaluacion-iCub-ELO308/train9_pc_icub/weights/last.pt")
nombre_archivo = 'datos_icub.csv'
yarp.Network.init()

def escribir_a_csv(datos):
    with open(nombre_archivo, 'a', newline='') as archivo_csv:
        escritor_csv = csv.writer(archivo_csv)
        escritor_csv.writerow(datos)

def calcular_diametro_desde_area(area):
    radio = math.sqrt(area / math.pi)
    diametro = 2 * radio
    return diametro


focal_length_x = 215.483
focal_length_y = 214.935
optical_center_x = 174.868
optical_center_y = 105.63
K = np.array([[focal_length_x, 0, optical_center_x],
              [0, focal_length_y, optical_center_y],
              [0, 0, 1]])


theta = (50)/(math.pi*180)

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

while(1):
    yarp_image = yarp.ImageRgb()
    yarp_image.resize(320, 240)
    yarp_image.setExternal(img_array, img_array.shape[1], img_array.shape[0])
    input_port.read(yarp_image)
    img_rgb = cv.cvtColor(img_array, cv.COLOR_BGR2RGB)

    #----------YOLO implementation-----------#

    results = model.track(img_rgb, conf=0.5, device=0, persist=True)

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
    coords_hom_robot = np.array([x, y, z, 1])

    coords_robot_cam = np.dot(T_inverse, coords_hom_robot)
    
    x_final = coords_robot_cam[0]
    y_final = coords_robot_cam[1]
    z_final = coords_robot_cam[2]
    datos = np.array([x_final, y_final, z_final])
    escribir_a_csv(datos)

    cv.putText(annotated_frame, f"Coordenadas: ({x_final}, {y_final}, {z_final})", (50,50),cv.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 1)
    cv.imshow("Imagen recibida", annotated_frame)
    print(f"({x_final}, {y_final}, {z_final}, {coords_robot_cam[3]})")



    if cv.waitKey(1) & 0xFF == ord('q'):
        break


cv.destroyAllWindows()
iPos.positionMove(home.data())