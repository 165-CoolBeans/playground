#=========================================================================================================#
# uses masking techniques to detect dark and light pollen separately
#   - defective, no contours detected
#   - possible causes: BGR value range incorrect AND/OR Canny edge detection threshold issue
#
# referenced from: 
#   - http://www.sixthresearcher.com/counting-blue-and-white-bacteria-colonies-with-python-and-opencv/
#=========================================================================================================#

import cv2 as cv
import numpy as np
import imutils as imutils

def smolShow(img, caption):
    scale_percent = 20
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)

    cv.imshow(caption, resized)

counter = {}

img = cv.imread("./input/pollen1.jpg")
w = img.shape[1]
h = img.shape[0]

imagecontour = img.copy()

# dictionary of pollen

pollen_types = ['dark', 'light']
for pollen_type in pollen_types:
    imgcpy = img.copy()

    counter[pollen_type] = 0

    if pollen_type == 'dark':
        lb = np.array([61, 40, 61])        # BGR values
        ub = np.array([64, 32, 63])
    elif pollen_type == 'light':
        imgcpy = (255-imgcpy)
        lb = np.array([71, 93, 65])
        ub = np.array([81, 96, 75])

    img_mask = cv.inRange(imgcpy, lb, ub)
    img_res = cv.bitwise_and(imgcpy, imgcpy, mask = img_mask)

    img_gray = cv.cvtColor(img_res, cv.COLOR_BGR2GRAY)
    img_gray = cv.GaussianBlur(img_gray, (5,5), 0)

    img_edge = cv.Canny(img_gray, 30, 35)          # adjust threshold
    img_edge = cv.dilate(img_edge, None, iterations = 1)
    img_edge = cv.erode(img_edge, None, iterations = 1)

    smolShow(imgcpy, "edges")
    cv.waitKey(0)

    # find contours
    contours = cv.findContours(img_edge.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if imutils.is_cv2() else contours[1]

    print(contours)

    for contour in contours:

        if cv.contourArea(contour) < 5:
            continue

        hull = cv.convexHull(contour)
        if pollen_type == 'dark':
            cv.drawContours(imagecontour, [hull], 0, (0,0,255),1)
        elif pollen_type == 'light':
            cv.drawContours(imagecontour, [hull], 0, (0,0,255),1)
        
        counter[pollen_type] += 1
    
    print("{} {} colonies".format(counter[pollen_type], pollen_type))

smolShow(img, str(w) + "(w) x " + str(h) + "(h)")
smolShow(imagecontour, "contours")
cv.waitKey(0)

cv.destroyAllWindows()