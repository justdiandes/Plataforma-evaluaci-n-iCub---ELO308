import csv
import rospy
from std_msgs.msg import Float64MultiArray

class SynchronizedData:
    def __init__(self, timestamp, sensor_name, data):
        self.timestamp = timestamp
        self.sensor_name = sensor_name
        self.data = data

def callback(sensor_data, sensor_name):
    # Agregar un retardo específico para cada sensor (en segundos)
    if sensor_name == 'cam':
        rospy.sleep(0.1)  # Retardo de 0.1 segundos
    elif sensor_name == 'icub':
        rospy.sleep(0.1)  # Retardo de 0.1

    # Procesar datos y guardar en un archivo CSV
    timestamp = rospy.Time.now()
    synchronized_data = SynchronizedData(timestamp, sensor_name, sensor_data)
    save_to_csv(synchronized_data)

def save_to_csv(synchronized_data):
    with open('datos.csv', 'a') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([synchronized_data.timestamp, synchronized_data.sensor_name] + list(synchronized_data.data.data))

# Inicializar el nodo ROS
rospy.init_node('data_saver')

# Utilizar Subscriber para suscribirse a los tópicos
sub1 = rospy.Subscriber('/datos_sensor1', Float64MultiArray, callback, callback_args='cam', queue_size=10)
sub2 = rospy.Subscriber('/datos_sensor2', Float64MultiArray, callback, callback_args='icub', queue_size=10)

rospy.loginfo("Nodo iniciado. Esperando datos...")

# Mantener el nodo en ejecución
rospy.spin()
