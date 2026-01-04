import numpy as np
import cv2

def calculate_area(image):
    height, width = image.shape[:2]

    b = image[:, :, 0]
    g = image[:, :, 1]
    r = image[:, :, 2]

    mask_red    = ((b == 0)   & (g == 0)   & (r == 255)).astype(np.uint8) * 255
    mask_green  = ((b == 0)   & (g == 255) & (r == 0)).astype(np.uint8) * 255
    mask_blue   = ((b == 255) & (g == 0)   & (r == 0)).astype(np.uint8) * 255
    mask_yellow = ((b == 0)   & (g == 255) & (r == 255)).astype(np.uint8) * 255
    mask_purple = ((b == 255) & (g == 0)   & (r == 255)).astype(np.uint8) * 255
    mask_gray   = ((b == g) & (g == r) & (r != 0)).astype(np.uint8) * 255

    masks = {
        "red": mask_red,
        "green": mask_green,
        "blue": mask_blue,
        "yellow": mask_yellow,
        "purple": mask_purple,
        "gray": mask_gray,
    }

    order = ["red", "green", "blue", "yellow", "purple", "gray"]
    result = {}
    total_shapes = 0

    for color in order:
        contours = cv2.findContours(
            masks[color], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[0]

        color_sum = 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area_base = cv2.contourArea(contour)

            if w == h:
                area = area_base
            else:
                area = area_base * 3

            color_sum += area
            total_shapes += 1

        if color_sum > 0:
            result[color] = int(color_sum)

    if total_shapes == 0:
        return f"black, {height * width}"

    lines = []
    for color in order:
        if color in result:
            lines.append(f"{color}, {result[color]}")

    return "\n".join(lines)
