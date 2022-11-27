import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


img1 = cv.imread('./input/pollen1.jpg')
img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2HSV)
img1_gray = cv.split(img1_gray)[2]
#img1_gray = cv.medianBlur(img1_gray, 5)

cv.imwrite("./output/output1.jpg", img1_gray)

hist = cv.calcHist(img1_gray, [0], None, [256], [0, 256])
mode = np.argmax(hist)
_, img2 = cv.threshold(img1_gray, 110, mode, cv.THRESH_BINARY)
_, img2 = cv.threshold(img1_gray, 110, 255, cv.THRESH_BINARY)

img2 = cv.erode(img2, np.ones((3, 3), np.uint8), iterations=10)
img2 = cv.dilate(img2, np.ones((3, 3), np.uint8), iterations=10)


cv.imwrite("./output/output2.jpg", img2)

rows = img1.shape[0]
circles = cv.HoughCircles(img2, cv.HOUGH_GRADIENT, 1, 
                            rows/15, param1=30, param2=10,
                            minRadius=10, maxRadius=200)


if circles is not None:
    print(len(circles[0]))
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        # circle center
        cv.circle(img1, center, 1, (0, 100, 100), 3)
        # circle outline
        radius = i[2]
        cv.circle(img1, center, radius, (255, 0, 255), 3) 

cv.imwrite("./output/output3.jpg", img1)