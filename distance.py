#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
import math

# Variables para almacenar las coordenadas de ambos objetos
x1, y1, z1 = 0, 0, 0
x2, y2, z2 = 0, 0, 0

def distance_callback(msg):
    global x1, y1, z1, x2, y2, z2

    # Si el mensaje es del primer objeto, actualiza sus coordenadas
    if msg.header.frame_id == 'red_ball':
        x1, y1, z1 = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
    # Si el mensaje es del segundo objeto, actualiza sus coordenadas
    elif msg.header.frame_id == 'camera':
        x2, y2, z2 = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z

    # Calcular la distancia euclidiana
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

    # Convertir la distancia a centímetros si es necesario
    distance_cm = distance * 100

    # Imprimir la distancia en centímetros
    rospy.loginfo("Distancia entre objetos: %.2f centímetros", distance_cm)

def main():
    rospy.init_node('distance_calculator', anonymous=True)
    
    # Sustituye 'objeto_1/pose' y 'objeto_2/pose' con los tópicos correctos de tus objetos
    rospy.Subscriber('red_ball', PoseStamped, distance_callback)
    rospy.Subscriber('camera', PoseStamped, distance_callback)

    rospy.spin()

if __name__ == '__main__':
    main()
