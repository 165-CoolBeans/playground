import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


img1 = cv.imread('./input/pollen1.jpg')
img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
#img1_gray = cv.split(img1_gray)[2]
#img1_gray = cv.medianBlur(img1_gray, 5)

cv.imwrite("./output/output1.jpg", img1_gray)

#plt.hist(img1_gray.ravel(), 256, [0, 256])
#plt.show()

clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl_img = clahe.apply(img1_gray)

cv.imwrite("./output/output2.jpg", cl_img)

_, img2 = cv.threshold(img1_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
_, img3 = cv.threshold(cl_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)


img2 = cv.erode(img2, np.ones((3, 3), np.uint8), iterations=10)
img2 = cv.dilate(img2, np.ones((3, 3), np.uint8), iterations=10)
cv.imwrite("./output/output3.jpg", img2)

rows = img1.shape[0]
circles = cv.HoughCircles(img2, cv.HOUGH_GRADIENT, 1, 
                            rows/24, param1=200, param2=10,
                            minRadius=40, maxRadius=75)

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

cv.imwrite("./output/output4.jpg", img1)
