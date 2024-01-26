import yarp
import numpy
import cv2 as cv


yarp.Network.init()

# Función para manejar la acción de la flecha izquierda
def accion_flecha_izquierda():
    print("Acción con Flecha Izquierda")
    #initialize a new tmp vector identical to encs
    tmp2=yarp.Vector(jnts)
    tmp2.set(0, tmp2.get(3)-40)
    iPos.positionMove(tmp2.data()) 

# Función para manejar la acción de la flecha derecha
def accion_flecha_derecha():
    print("Acción con Flecha Derecha")
    #initialize a new tmp vector identical to encs
    tmp3=yarp.Vector(jnts)
    tmp3.set(0, tmp3.get(4)-40)
    iPos.positionMove(tmp3.data()) 


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




while True:
    yarp_image = yarp.ImageRgb()
    yarp_image.resize(320, 240)
    yarp_image.setExternal(img_array, img_array.shape[1], img_array.shape[0])
    input_port.read(yarp_image)
    img_rgb = cv.cvtColor(img_array, cv.COLOR_BGR2RGB)

    # Muestra la imagen en una ventana llamada "Imagen"
    



    # Verifica la tecla presionada
    if cv.waitKey(1) & 0xFF == 27:  # 27 es el código ASCII para la tecla Esc
        break
    elif cv.waitKey(1) & 0xFF == ord('a'):  # Código ASCII para la flecha izquierda
        accion_flecha_izquierda()
    elif cv.waitKey(1) & 0xFF == ord('d'):  # Código ASCII para la flecha derecha
        accion_flecha_derecha()
    cv.imshow("Imagen", img_rgb)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    # Cierra la ventana al salir del bucle
cv.destroyAllWindows()
iPos.positionMove(home.data())