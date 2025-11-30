# coding: utf-8

import cv2
import numpy as np
import math
def detect_clock(binary_image):
    if len(binary_image.shape) == 3:
        gray = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            21,
            3
        )
    else:
        gray = binary_image
        binary = gray   # 🔥 بسیار مهم

    circles = cv2.HoughCircles(
        binary,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=300,
        param1=100,
        param2=20,
        minRadius=120,
        maxRadius=260
    )

    if circles is None:
        # اگر دایره‌ای پیدا نشد، تصویر ورودی را برگردان
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    circles = np.uint16(np.around(circles))
    x, y, r = circles[0][0]
    center = (x, y)

    # لبه یابی برای پیدا کردن خطوط
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)

    # پیدا کردن خطوط با HoughLinesP
    lines = cv2.HoughLinesP(edges,
                            rho=1,
                            theta=np.pi / 180,
                            threshold=100,
                            minLineLength=40,
                            maxLineGap=10)
    if lines is None:
        # اگر خطی پیدا نشد، فقط دایره را رسم کن و بازگردان
        new_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.circle(new_image, (x, y), r, (0, 255, 0), 3)
        return new_image

    valid_lines = []
    x_center, y_center = x, y

    for line in lines:
        x1, y1, x2, y2 = line[0]

        d1 = np.sqrt((x1 - x_center) ** 2 + (y1 - y_center) ** 2)
        d2 = np.sqrt((x2 - x_center) ** 2 + (y2 - y_center) ** 2)

        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # فیلتر کردن خطوط کوتاه
        if length < 50:
            continue

        # حداقل یکی از نقاط باید به مرکز نزدیک باشد
        if d1 > 40 and d2 > 40:
            continue

        angle = np.arctan2(y2 - y1, x2 - x1)
        valid_lines.append((line[0], length, angle))

    if len(valid_lines) < 2:
        # اگر کمتر از دو خط معتبر پیدا شد، فقط دایره را رسم کن
        new_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.circle(new_image, (x, y), r, (0, 255, 0), 3)
        return new_image

    # مرتب سازی خطوط بر اساس طول به صورت نزولی
    valid_lines.sort(key=lambda x: x[1], reverse=True)

    # حذف خطوط موازی (زاویه نزدیک به هم)
    filtered = []
    for coords, length, angle in valid_lines:
        if any(abs(angle - a) < 0.1 for (_, _, a) in filtered):
            continue
        filtered.append((coords, length, angle))

    if len(filtered) < 2:
        # اگر کمتر از دو خط پس از حذف موازی‌ها ماند، همان خطوط را استفاده کن
        filtered = valid_lines[:2]

    # انتخاب دو خط بلندترین (دقیقه) و دومین بلند (ساعت)
    line_minute, line_hour = filtered[0][0], filtered[1][0]

    # رسم خطوط و دایره روی تصویر رنگی
    new_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    x1, y1, x2, y2 = line_minute
    cv2.line(new_image, (x1, y1), (x2, y2), (0, 255, 0), 3)  # سبز دقیقه

    x1, y1, x2, y2 = line_hour
    cv2.line(new_image, (x1, y1), (x2, y2), (0, 0, 255), 3)  # قرمز ساعت

    cv2.circle(new_image, (x, y), r, (0, 255, 0), 3)  # دایره سبز

    return line_hour, line_minute, center
