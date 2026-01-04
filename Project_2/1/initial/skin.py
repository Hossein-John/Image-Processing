import cv2
import numpy as np

def detect_skin(image_path):
    # خواندن تصویر (BGR)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found")

    # تبدیل به HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ----------- بازه‌ی پوست (تجربی و پایدار) -----------
    lower_skin = np.array([0, 40, 60])
    upper_skin = np.array([25, 180, 255])

    # ماسک اولیه
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # ----------- عملیات مورفولوژیکی -----------
    kernel = np.ones((5, 5), np.uint8)

    # حذف نویز
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # پر کردن صورت
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # ----------- خروجی RGB سیاه‌وسفید -----------
    output = np.zeros_like(img)
    output[mask == 255] = [255, 255, 255]

    return output

