import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import cvzone

from cvzone.ColorModule import ColorFinder
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_point(ax, x, y, z):
    ax.scatter(x, y, z, c='r', marker='o')
    plt.draw()

KNOWN_DISTANCE = 30 #centimeters
KNOWN_WIDTH = 6.2 #centimeters

def Focal_length(measured_distance, real_width, width_in_rf_image):
    focal_length = (width_in_rf_image * measured_distance)/real_width
    return focal_length

def Distance_finder(Focal_Length, real_object_width, object_width_in_frame):    
    distance = (real_object_width * Focal_Length)/object_width_in_frame
    return distance


#-----------Revisión de la cámara disponible----------------#




class CameraSubscriber:
    def __init__(self):
        rospy.init_node('camera_subscriber', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/my_camera/image_raw', Image, self.image_callback)
        self.video_capture = cv2.VideoCapture()

    def image_callback(self, msg):
        try:
            # Convierte el mensaje de imagen a una imagen OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Procesa el frame aquí, puedes mostrarlo con cv2.imshow o hacer cualquier otra operación
            cv2.imshow('Imagen de la cámara', cv_image)
            
            # Si deseas guardar el video, puedes hacerlo con cv2.VideoWriter

            # Espera 1ms y verifica si se presiona la tecla 'q' para salir
            if cv2.waitKey(1) & 0xFF == ord('q'):
                rospy.signal_shutdown('Usuario presionó "q"')

        except Exception as e:
            print(e)

    def run(self):
        rate = rospy.Rate(10)  # Puedes ajustar la frecuencia de actualización según sea necesario
        while not rospy.is_shutdown():
            ret, frame = self.video_capture.read()
            rate.sleep()

if __name__ == '__main__':
    try:
        camera_subscriber = CameraSubscriber()
        rate = rospy.Rate(30)  # Puedes ajustar la frecuencia de actualización según sea necesario
        
        ref_image = cv2.imread("conocida30cm.png")

        colorF = ColorFinder(False)
        #Hue, saturation, value
        #hsvVals = {'hmin': 0, 'smin': 91, 'vmin': 124, 'hmax': 6, 'smax': 255, 'vmax': 255}# Este es el primero, funciona con menos luminosidad
        #Buscar la forma de tener un rango de detección
        hsvVals = {'hmin': 0, 'smin': 74, 'vmin': 117, 'hmax': 179, 'smax': 255, 'vmax': 255}
        #--------------Matríz de parámetros intrínsecos-------------------#
        focal_length_x = 823.2638336
        focal_length_y = 830.00741099
        optical_center_x = 320.55874298
        optical_center_y = 305.66295207
        K = np.array([[focal_length_x, 0, optical_center_x],
                    [0, focal_length_y, optical_center_y],
                    [0, 0, 1]])

        #--------------Extracción datos desde imagen referencia para cálculo de distancia------------------#

        imgColoRef, maskRef = colorF.update(ref_image, hsvVals)
        imgContourRef, contoursRef = cvzone.findContours(ref_image, maskRef, minArea=5000)
        href, wref, _ = ref_image.shape
        if contoursRef:
            data = contoursRef[0]['center'][0],\
                    href - contoursRef[0]['center'][1],\
                    int(contoursRef[0]['area'])

        cv2.imshow("Ref", imgContourRef)
        object_width_image = 2*(int(contoursRef[0]['area']))/((np.pi)**2)
        Focal_length_found = Focal_length(KNOWN_DISTANCE, KNOWN_WIDTH, object_width_image)

        Base_distance = 50 #cm


        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlim([-20, 20])
        ax.set_ylim([-20, 20])
        ax.set_zlim([-20, 20])

        
        while not rospy.is_shutdown():
            ret, frame = camera_subscriber.video_capture.read()
            rate.sleep()

            h, w,_ = frame.shape

            imgColor, mask = colorF.update(frame, hsvVals)
            #--------Función cvzone permite obtener los bounding boxes automáticamente------#
            imgContour, contours = cvzone.findContours(frame, mask, minArea=5000) #Utiliza cv2 para ser creada pero cvzone automatiza este proceso

            if contours: #aquí se obtiene la coordenada del punto central del bounding box en (x,y) y se obtiene el parámetro de área del bounding box
                data = contours[0]['center'][0],\
                        h - contours[0]['center'][1], \
                        int(contours[0]['area']) #Lista de contornos -> Queremos el controrno más grande
                #print(data)

            
            #both2 = np.concatenate((mask, imgContour), axis=1)
            if data[2] != 0:
                Distance = Distance_finder(Focal_length_found, KNOWN_WIDTH, 2*(data[2])/((np.pi)**2))
            else:
                Distance = 0
            #--------------------------Cálculo de coordenadas--------------------------------#
            z_coordinate = Base_distance - Distance
            u_frame = data[0]
            v_frame = data[1]
            
            #proyectar puntos 2D en plano imagen
            image_plane_coords = np.array([[u_frame], [v_frame], [1]])
            #Normalizar coordenadas 2D
            normalized_coords = np.dot(np.linalg.inv(K), image_plane_coords)
            #Cálculo coordenadas 3D
            #object_coords = image_plane_coords
            object_coords = normalized_coords * z_coordinate
            object_coords = object_coords.round(5)
            #object_coords = normalized_coords
            print(f"({object_coords.item(0)}, {object_coords.item(1)}, {object_coords.item(2)})")

            cv2.putText(imgContour, f"Coordenadas: ({object_coords.item(0)}, {object_coords.item(1)}, {object_coords.item(2)})", (50,50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 1)
            cv2.putText(imgContour, f"Distancia [cm] = {Distance}",(50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(imgContour, f"Distancia [cm] = {z_coordinate}",(50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            both1 = np.concatenate((frame, imgColor), axis=1)



            cv2.imshow("Image color", both1)
            cv2.imshow("Color detection", imgContour)

            x = object_coords.item(0)
            y = object_coords.item(1)
            z = object_coords.item(2)

            # Borra el gráfico anterior
            ax.cla()

            ax.set_xlim([-20, 20])
            ax.set_ylim([-20, 20])
            ax.set_zlim([-20, 20])
            # Plotea el nuevo punto
            plot_point(ax, x, y, z)


            plt.pause(0.1)
            
            #cv2.imshow("Image frame o", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            cv2.release()
            cv2.destroyAllWindows()




    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()