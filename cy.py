import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# util fxn; scale output to fit in screen
def smolShow(img, caption):
    scale_percent = 20                               # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)

    cv.imshow(caption, resized)

# read image
img = cv.imread('./input/pollen1.jpg')

# grayscale image
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# remove noise by using Gaussian blur
img_blur = cv.GaussianBlur(img_gray, (51,51), 0)

# use canny edge detection to get object edges; stuck here; outputs nothing on img_blur input
img_canny = cv.Canny(img_blur, 0, 2000)

# show images
smolShow(img, "default")
smolShow(img_gray, "grayscaled")
smolShow(img_blur, "blurred")
smolShow(img_canny, "canny")

cv.waitKey(0)
cv.destroyAllWindows()

# save image
#cv.imwrite('./outbin/gray.jpg', img_gray)
#cv.imwrite('./outbin/blur.jpg', img_blur)
cv.imwrite('./outbin/canny.jpg', img_canny)

print("Operation Done")