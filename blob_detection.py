import numpy as np
import cv2 as cv

fp = open("output.txt", "w")

def getGreenChannel(img):
    output = img.copy()
    output[:,:,0] = 0
    output[:,:,2] = 0

    return output

img1 = cv.imread('./input/pollen1.jpg')
img1_green = getGreenChannel(img1)
img1_gray = cv.cvtColor(img1_green, cv.COLOR_BGR2GRAY)

cv.imwrite("./output/output1.jpg", img1_gray)

params = cv.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 4000
params.maxArea = 15000

params.filterByCircularity = True
params.minCircularity = 0.3

params.filterByInertia = True
params.minInertiaRatio = 0.25

params.filterByConvexity = True
params.minConvexity = 0.5


detector = cv.SimpleBlobDetector_create(params)

keypoints = detector.detect(img1_gray)

img2 = cv.drawKeypoints(img1_gray, keypoints, np.zeros((1, 1)), (0, 0, 255), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

print(f"Total detected: {len(keypoints)}")
cv.imwrite("./output/output2.jpg", img2)