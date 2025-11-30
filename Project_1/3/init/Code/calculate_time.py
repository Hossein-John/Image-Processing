# coding: utf-8

import cv2
import numpy as np
import math
def calculate_time(line_hour, line_minute, center):
    
    cx, cy = center
    x1h, y1h, x2h, y2h = line_hour
    x1m, y1m, x2m, y2m = line_minute

    # ---------- پیدا کردن far point عقربه ساعت ----------
    d1 = np.sqrt((x1h - cx)**2 + (y1h - cy)**2)
    d2 = np.sqrt((x2h - cx)**2 + (y2h - cy)**2)

    if d1 > d2:
        far_hour = (x1h, y1h)
    else:
        far_hour = (x2h, y2h)

    # ---------- پیدا کردن far point عقربه دقیقه ----------
    d1 = np.sqrt((x1m - cx)**2 + (y1m - cy)**2)
    d2 = np.sqrt((x2m - cx)**2 + (y2m - cy)**2)

    if d1 > d2:
        far_minute = (x1m, y1m)
    else:
        far_minute = (x2m, y2m)

    # ---------- زاویه عقربه ساعت ----------
    dx = far_hour[0] - cx
    dy = far_hour[1] - cy
    angle_hour = np.degrees(np.arctan2(dy, dx))
    angle_hour = (angle_hour + 90) % 360     # تبدیل به مختصات ساعت

    # ---------- زاویه عقربه دقیقه ----------
    dx = far_minute[0] - cx
    dy = far_minute[1] - cy
    angle_minute = np.degrees(np.arctan2(dy, dx))
    angle_minute = (angle_minute + 90) % 360

    # ---------- تبدیل زاویه به ساعت ----------
    hour = int(angle_hour // 30)             # هر 30 درجه = 1 ساعت
    if hour == 0:
        hour = 12

    # ---------- تبدیل زاویه به دقیقه ----------
    minute = int(angle_minute // 6)          # هر 6 درجه = 1 دقیقه
    minute = round(minute / 5) * 5           # گرد کردن دقیقه

    # دقیقه 60 → ساعت +1
    if minute == 60:
        minute = 0
        hour += 1
        if hour == 13:
            hour = 1

    return f"{hour:02d}:{minute:02d}"
