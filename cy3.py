import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def smolShow(img, caption):
    scale_percent = 40
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)

    cv.imshow(caption, resized)

img = cv.imread('./input/pollen1.jpg')

gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray_img = cv.GaussianBlur(gray_img, (7,7), 0)

kernel = np.ones((3,3), np.uint8)
gray_img = cv.morphologyEx(gray_img, cv.MORPH_OPEN, kernel)

canny_img = cv.Canny(gray_img, 30, 35)

height = img.shape[0]
width = img.shape[1]

pollen = cv.HoughCircles(canny_img, cv.HOUGH_GRADIENT, 1, height/256, param1=35, param2=100, minRadius=1, maxRadius=750)

if pollen is not None:
    pollen = np.uint16(np.around(pollen))
    for i in pollen[0, :]:
        center = (i[0], i[1])
        cv.circle(img, center, 1, (0,100,100), 3)
        radius = i[2]
        cv.circle(img, center, radius, (255,0,255), 3)

smolShow(img, "zamnn")
cv.waitKey(0)

cv.destroyAllWindows()