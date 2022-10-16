import numpy as np
import cv2 as cv

def showImage(img, name="Image"):
    cv.imshow(name, img)
    cv.waitKey(0)

def scaleImage(img, scale=0.75):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)

    return cv.resize(img, (width, height))

def getImageMidpoint(img): return (img.shape[1] // 2, img.shape[0] // 2)


img1 = cv.imread('./input/coins1.jpg', cv.IMREAD_GRAYSCALE)

_, img2 = cv.threshold(img1, 130, 255, cv.THRESH_BINARY)

cv.imwrite("./output/output1.jpg", img2)


dilate = cv.erode(img2, (4, 4), iterations=3)
cv.imwrite("./output/output_dilate.jpg", dilate)

erode = cv.dilate(img2, (4, 4), iterations=3)
cv.imwrite("./output/output_erode.jpg", erode)

img3 = cv.Canny(dilate, 50, 150)

cv.imwrite("./output/output2.jpg", img3)

cv.waitKey(0)