import cv2 as cv
import numpy as np

def smolShow(img, caption):
    scale_percent = 20
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)

    cv.imshow(caption, resized)

# params
blur_kernel = (7, 7)
canny_lb = 30
canny_up = 35

img = cv.imread('./input/pollen1.jpg')
cimg = img.copy()

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

blur = cv.GaussianBlur(gray, blur_kernel, 0)

edge = cv.Canny(blur, canny_lb, canny_up)

# hough
circles = cv.HoughCircles(
                            blur,
                            cv.HOUGH_GRADIENT,
                            1,                          # inverse ratio resolution
                            35,                         # minimum distance between circle centers
                            param1=30,                  # upper threshold for Canny
                            param2=20,                  # threshold for center detection; lower = more circles but prone to false circles
                            minRadius=20,               # minimum circle radius
                            maxRadius=55                # max radius
                        )

circles = np.uint16(np.around(circles))

for i in circles[0,:]:
    cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),4)
    cv.circle(cimg,(i[0],i[1]),2,(0,0,255),2)

# show
smolShow(img, "orig")
smolShow(blur, "blur")
smolShow(edge, "edge")
smolShow(cimg, "final")

cv.waitKey(0)
cv.destroyAllWindows()