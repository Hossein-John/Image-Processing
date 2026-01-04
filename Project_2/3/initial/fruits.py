import cv2
import numpy as np

def create_mask(hsv_image, hsv_lower, hsv_upper):
    """Create a mask with given hsv threshold values"""
    mask = cv2.inRange(hsv_image, hsv_lower, hsv_upper)
    return mask

def get_rect(mask):
    """
    Finds contours on a given mask, finds rectangles around these contours
    and returns dimensions of found_objects rectangles
    """
    conts, _hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    found_objects = 0

    for cont in conts:
        rect = cv2.boundingRect(cont)
        x, y, w, h = rect

        if w < 70 or h < 70:
            continue

        rects.append(rect)
        found_objects += 1

    return rects, found_objects

def detect_fruits(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return [0, 0, 0]
        
    img_res = cv2.resize(img, dsize=None, fx=0.15, fy=0.15, interpolation=cv2.INTER_CUBIC)
    img_hsv = cv2.cvtColor(img_res, cv2.COLOR_BGR2HSV)

    # Apple mask - using a single range (you might need to combine masks for different apple colors)
    apple_lower_1 = np.array([0, 50, 40])
    apple_upper_1 = np.array([9, 220, 255])
    apple_mask_1 = create_mask(img_hsv, apple_lower_1, apple_upper_1)

    apple_lower_2 = np.array([0,  75,  75])
    apple_upper_2 = np.array([18, 215, 150])
    apple_mask_2 = create_mask(img_hsv, apple_lower_2, apple_upper_2)

    apple_lower_3 = np.array([130,  25,  40])
    apple_upper_3 = np.array([180, 200, 230])
    apple_mask_3 = create_mask(img_hsv, apple_lower_3, apple_upper_3)

    combined_apple_mask = cv2.bitwise_or(apple_mask_1, apple_mask_2)
    apple_mask = cv2.bitwise_or(combined_apple_mask, apple_mask_3)

    # Banana mask
    banana_lower = np.array([20, 90, 120])
    banana_upper = np.array([30, 255, 250])
    banana_mask = create_mask(img_hsv, banana_lower, banana_upper)

    # Orange mask
    orange_lower = np.array([10, 200, 130])
    orange_upper = np.array([18, 255, 255])
    orange_mask = create_mask(img_hsv, orange_lower, orange_upper)

    apple_rects, apple_found_objects = get_rect(apple_mask)
    banana_rects, banana_found_objects = get_rect(banana_mask)
    orange_rects, orange_found_objects = get_rect(orange_mask)

    return [apple_found_objects, banana_found_objects, orange_found_objects]
    