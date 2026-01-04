import cv2
from calculate_area import calculate_area


img = cv2.imread("Morphology/2/sample/new2.png")  
cv2.imshow("Input Image", img)
cv2.waitKey(0)

result = calculate_area(img)
print(result)
