import yarp
import numpy
import cv2 as cv
from ultralytics import YOLO

model = YOLO("/home/diego/Escritorio/Plataforma-evaluacion-iCub-ELO308/train9_pc_icub/weights/last.pt")

yarp.Network.init()


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


    #----------------------------------------#


    cv.imshow("yarp", annotated_frame)




    if cv.waitKey(1) & 0xFF == ord('q'):
        break


cv.destroyAllWindows()
iPos.positionMove(home.data())