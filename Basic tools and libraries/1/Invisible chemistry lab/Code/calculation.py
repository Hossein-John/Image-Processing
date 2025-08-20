# coding: utf-8

import cv2
import numpy as np
def calculation(image):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([90, 100, 50])
    upper_blue = np.array([140, 255, 255])

    mask = cv2.inRange(img_hsv, lower_blue, upper_blue)

    blue_pix = cv2.countNonZero(mask)

    tube_height = 200
    tube_width = 60
    num_tubes = 3
    total_tube_area = num_tubes * tube_height * tube_width

   
    average_filled_percentage = (blue_pix / total_tube_area) * 100

    return int(average_filled_percentage)
