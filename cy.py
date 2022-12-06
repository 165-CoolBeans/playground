import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("./input/scoot.jpg")

# img.shape [height, width]

width = int(img.shape[1] * 40/100)
height = int(img.shape[0] * 40/100)

img2 = cv.resize(img, (width, height), interpolation = cv.INTER_AREA)

cv.imshow(str(img2.shape), img2)
cv.waitKey(0)

cv.imwrite("./output/_out.jpg", img2)