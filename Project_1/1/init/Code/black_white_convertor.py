# coding: utf-8

import cv2
import numpy as np
def black_white_convertor(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    background_pixels = np.argmax(hist)
    mask = np.where(gray_image == background_pixels, 255, 0).astype(np.uint8)

    return mask
