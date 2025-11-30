# coding: utf-8

import cv2
import numpy as np
import math
def black_white_convertor(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    new_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 3)
    
    return new_image
