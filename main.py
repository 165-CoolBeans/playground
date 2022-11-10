import numpy as np
import cv2 as cv

img1 = cv.imread('./input/pollen1.jpg')
img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

#   binarize grayscale image
_, img2 = cv.threshold(img1_gray, 130, 255, cv.THRESH_BINARY_INV)
mask = np.zeros(img1_gray.shape, dtype="uint8")

cv.imwrite("./output/output1.jpg", img2)