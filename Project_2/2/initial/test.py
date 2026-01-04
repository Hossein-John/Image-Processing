import cv2
import numpy as np

def detect_player(image_path):
    # خواندن تصویر (BGR)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found")

    # تبدیل به HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

  
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    # ماسک اولیه
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.bitwise_not(mask)

    # ----------- عملیات مورفولوژیکی -----------
    kernel = np.ones((3, 3), np.uint8)

    # حذف نویز
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # پر کردن صورت
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=10)


    

    return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)