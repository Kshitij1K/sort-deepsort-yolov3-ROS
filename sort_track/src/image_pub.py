#!/usr/bin/env python

import rospy
import numpy as np
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import Image
from os import path

if __name__ == "__main__":
        rospy.init_node('sort_tracker', anonymous=False)
        image_pub = rospy.Publisher("usb_cam/image_raw",Image,queue_size=10)
        cap = cv2.VideoCapture(path.expanduser('~')+'/Downloads/DJI_0012.MP4')
        bridge = CvBridge()
        loopRate = rospy.Rate(15)
        while (not rospy.is_shutdown()):
                ret, frame = cap.read()
                bridge = CvBridge()
                if np.shape(frame) == ():
                        cap.release()
                        cap = cv2.VideoCapture(path.expanduser('~')+'/Downloads/DJI_0012.MP4')
                else:
                        image_msg = bridge.cv2_to_imgmsg(frame,"bgr8")
                        image_pub.publish(image_msg) 
                loopRate.sleep()
