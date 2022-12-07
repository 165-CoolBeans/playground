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
img_blur = cv.GaussianBlur(img_gray, (7,7), 0)

# use canny edge detection to get object edges
img_canny = cv.Canny(img_blur, 30, 35)

# morphological operations
kernel = np.ones((5,5), np.uint8)
#img_erode = cv.erode(img_canny, kernel, iterations = 1)
img_dilate = cv.dilate(img_canny, kernel, iterations = 1)
#img_open = cv.morphologyEx(img_canny, cv.MORPH_OPEN, kernel)
#img_close = cv.morphologyEx(img_canny, cv.MORPH_CLOSE, kernel)

# find and draw edges/contours
count, hierarchy = cv.findContours(img_dilate.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.drawContours(rgb, count, -1, (0, 0, 255), 2)

# show images
smolShow(img, "default")
smolShow(img_gray, "grayscaled")
smolShow(img_blur, "blurred")
smolShow(img_canny, "canny")
#smolShow(img_erode, "eroded")
smolShow(img_dilate, "dilated")
#smolShow(img_close, "closing")
#smolShow(img_open, "opening")
smolShow(rgb, "contours")

cv.waitKey(0)
cv.destroyAllWindows()

# print counts
print("Number of pollen detected: ", len(count))        # pollen1 manual count = 95, pollen2 = 113

# save image
#cv.imwrite('./outbin/gray.jpg', img_gray)
#cv.imwrite('./outbin/blur.jpg', img_blur)
#cv.imwrite('./outbin/canny.jpg', img_canny)

print("Operation Done")