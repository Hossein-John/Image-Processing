# coding: utf-8

import cv2
import numpy as np
def color_recognition(image):

    # --- normalize_brightness (inline) ---
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v = clahe.apply(v)

    normal_image = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
    # ------------------------------------

    # gray + otsu (فقط برای کمک)
    gray = cv2.cvtColor(normal_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    h, w = binary.shape

    # ROI بدنه
    x1, x2 = int(0.2 * w), int(0.8 * w)
    y1, y2 = int(0.35 * h), int(0.75 * h)

    # ROI رنگی (نه باینری!)
    roi_color = binary[y1:y2, x1:x2]
    white_pixels = np.sum(roi_color == 255)
    black_pixels = np.sum(roi_color == 0)
    

    return "white" if white_pixels > black_pixels else "black"
