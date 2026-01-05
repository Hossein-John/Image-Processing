# coding: utf-8

import cv2
import numpy as np
def coin_counter(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Apply morphological operations to remove noise and separate coins
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.erode(binary, kernel, iterations=1)
    binary = cv2.dilate(binary, kernel, iterations=3)

    # Find contours (only external ones)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by area to remove noise
    min_area = 4000  # Adjust this based on your coin size
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    
    return len(filtered_contours)
