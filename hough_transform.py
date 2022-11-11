import numpy as np
import cv2 as cv


img1 = cv.imread('./input/pollen1.jpg')
img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img1_gray = cv.medianBlur(img1_gray, 5)


rows = img1.shape[0]
circles = cv.HoughCircles(img1_gray, cv.HOUGH_GRADIENT, 1, 
                            rows/8, param1=50, param2=30,
                            minRadius=1, maxRadius=250)

print(circles)
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        # circle center
        cv.circle(img1, center, 1, (0, 100, 100), 3)
        # circle outline
        radius = i[2]
        cv.circle(img1, center, radius, (255, 0, 255), 3) 

cv.imwrite("./output/output3.jpg", img1)