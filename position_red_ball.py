import rospy
from std_msgs.msg import Float64MultiArray

from gazebo_msgs.srv import GetModelState
from geometry_msgs.msg import Pose
import time

pub = rospy.Publisher('/datos_sensor2', Float64MultiArray, queue_size=10)

def escribir_a_ros(datos):
    # Publicar datos en el tópico de ROS
    pub.publish(Float64MultiArray(data=datos))

def get_model_position(model_name):
    rospy.wait_for_service('/gazebo/get_model_state')
    try:
        get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        model_state = get_model_state(model_name, 'world')
        position = model_state.pose.position
        return [position.x, position.y, position.z]
    except rospy.ServiceException as e:
        print("Service call failed:", e)

if __name__ == '__main__':
    rospy.init_node('get_and_publish_model_position_node', anonymous=True)
    model_name = 'red-ball'  # Nombre del objeto simulado
    
    rate = rospy.Rate(30)  # Frecuencia de actualización en Hz (5 veces por segundo, para que coincida con el otro nodo)

    while not rospy.is_shutdown():
        position = get_model_position(model_name)
        print("Position of {}: {}".format(model_name, position))
        
        # Publicar las coordenadas en el mismo formato que el otro nodo
        datos = Float64MultiArray(data=position)
        escribir_a_ros(position)
        
        rate.sleep()
