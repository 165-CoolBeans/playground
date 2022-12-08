# was meant to use watershedding to detect overlapping pollen
# but highlights dark ones instead
# main ref: https://docs.opencv.org/3.3.1/d3/db4/tutorial_py_watershed.html

import cv2 as cv
import numpy as np

def smolShow(img, caption):
    scale_percent = 20
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)

    cv.imshow(caption, resized)

img = cv.imread('./input/pollen2.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

kernel = np.ones((3,3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations = 2)

sure_bg = cv.dilate(opening, kernel, iterations = 3)

dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
ret, sure_fg = cv.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)

ret, markers = cv.connectedComponents(sure_fg)

markers = markers + 1

markers[unknown==255] = 0

cimg = img.copy()

markers = cv.watershed(cimg, markers)
cimg[markers == -1] = [0, 0, 255]

smolShow(img, "orig")
smolShow(gray, "gray")
smolShow(thresh, "threshold")
smolShow(cimg, "tada")

cv.waitKey(0)
cv.destroyAllWindows()

#cv.imwrite("./output/darkpollen2.jpg", cimg)