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


img1 = cv.imread('input/fruit.png', cv.IMREAD_GRAYSCALE)

_, img2 = cv.threshold(img1, 150, 255, cv.THRESH_BINARY)


img2 = cv.erode(img2, (3, 3), iterations=3)
img2 = cv.dilate(img2, (3, 3), iterations=3)

img3 = cv.Canny(img2, 50, 150)

cv.imshow("Img1", img1)
cv.imshow("Img2", img2)
cv.imshow("Img3", img3)

cv.waitKey(0)