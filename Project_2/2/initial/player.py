import cv2
import numpy as np


def detect_player(address):
    image = cv2.imread(address)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    res = cv2.bitwise_and(image, image, mask=mask)
    res_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.threshold(res_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)



