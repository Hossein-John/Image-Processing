import cv2
import numpy as np
import os

TEMPLATE_FOLDER = "numbers/"

def process_plate(image):
    # Convert to grayscale and blur for noise reduction
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find license plate by aspect ratio and size
    plate = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            if 2.5 < w/h < 5 and 255 < w < 500 and 65 < h < 120:
                plate = image[y:y+h, x:x+w]
                break
    if plate is None:
        return None

    # Convert to binary for digit extraction
    gray_plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Extract number contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    numbers = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(cnt)
        if 0.3 <= aspect_ratio <= 0.8 and area > 5 and h > binary.shape[0] * 0.4:
            numbers.append((x, y, w, h, binary[y:y+h, x:x+w]))

    # Sort numbers left to right
    numbers.sort(key=lambda x: x[0])

    # Load templates
    templates = {}
    for filename in os.listdir(TEMPLATE_FOLDER):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            digit = filename.split(".")[0]
            img = cv2.imread(os.path.join(TEMPLATE_FOLDER, filename), cv2.IMREAD_GRAYSCALE)
            _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
            templates[digit] = img

    # Recognize numbers
    recognized = []
    scores = []
    for x, y, w, h, digit_img in numbers:
        best_match, best_score = None, float('inf')
        for digit, template in templates.items():
            template_resized = cv2.resize(template, (digit_img.shape[1], digit_img.shape[0]))
            diff = np.sum((digit_img.astype("float") - template_resized.astype("float")) ** 2)
            if diff < best_score:
                best_score, best_match = diff, digit

        if best_match is not None:
            scores.append(best_score)
            recognized.append((best_match, best_score))

    # Outlier filtering
    if scores:
        median_score = np.median(scores)
        max_allowed_score = 1.5 * median_score
        recognized = [int(digit) for digit, score in recognized if score <= max_allowed_score]

    # Limit to first five digits
    recognized = recognized[:5]

    # Decision logic
    if recognized:
        average = sum(recognized) / len(recognized)
        return 'True' if average > 6.5 else 'False'
    
    return 'False'
