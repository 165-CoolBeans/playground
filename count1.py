# source: https://pyimagesearch.com/2021/02/22/opencv-connected-component-labeling-and-analysis/

import numpy as np
import cv2 as cv

#   Utility functions
def showImage(img, name="Image"):
    cv.imshow(name, img)
    cv.waitKey(0)

def scaleImage(img, scale=0.75):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)

    if (scale > 1.0):
        return cv.resize(img, (width, height), interpolation=cv.INTER_LINEAR)
    else:
        return cv.resize(img, (width, height), interpolation=cv.INTER_AREA)

def getImageMidpoint(img): return (img.shape[1] // 2, img.shape[0] // 2)


img1 = cv.imread('./input/coins1.jpg')
img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

#   binarize grayscale image
_, img2 = cv.threshold(img1_gray, 130, 255, cv.THRESH_BINARY_INV)
mask = np.zeros(img1_gray.shape, dtype="uint8")

cv.imwrite("./output/output1.jpg", img2)

#   dilate to fill the gaps
kernel = np.ones((3, 3), np.uint8)
img2 = cv.dilate(img2, kernel, iterations=3)

#   output the inversion of img2
cv.imwrite("./output/output2.jpg", ~img2)

#   analyze connected components with 8 piece connectivity
totalLabels, labels, stats, centroid = cv.connectedComponentsWithStats(img2, 8, cv.CV_32S)
count = 0

#   check detected components
for i in range(1, totalLabels):
    x = stats[i, cv.CC_STAT_LEFT]
    y = stats[i, cv.CC_STAT_TOP]
    w = stats[i, cv.CC_STAT_WIDTH]
    h = stats[i, cv.CC_STAT_HEIGHT]
    area = stats[i, cv.CC_STAT_AREA]

    #   look for components that fit the area
    if area > 12000 and area < 13000:
        #   filters detected label from the mask
        componentMask = (labels == i).astype("uint8") * 255
        mask = cv.bitwise_or(mask, componentMask)

        #   draw a bounding rectangle on detected component
        cv.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 3)
        count += 1


cv.imwrite("./output/output3.jpg", ~mask)
cv.imwrite("./output/output4.jpg", img1)

print(f"There are {count} detected out of {totalLabels - 1} coins.")