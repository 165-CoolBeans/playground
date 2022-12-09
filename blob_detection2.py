import numpy as np
import cv2 as cv


img1 = cv.imread('./input/pollen1.jpg')
img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)


cv.imwrite("./output/output1.jpg", img1_gray)

clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl_img = clahe.apply(img1_gray)

cv.imwrite("./output/output2.jpg", cl_img)

_, img2 = cv.threshold(img1_gray, 60, 255, cv.THRESH_BINARY)
_, img3 = cv.threshold(cl_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)


img3 = cv.erode(img2, np.ones((3, 3), np.uint8), iterations=10)
img3 = cv.dilate(img2, np.ones((3, 3), np.uint8), iterations=10)
cv.imwrite("./output/output3.jpg", img2)

params = cv.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 1000
params.maxArea = 15000

params.filterByCircularity = True
params.minCircularity = 0.3

params.filterByInertia = True
params.minInertiaRatio = 0.25

params.filterByConvexity = True
params.minConvexity = 0.5


detector = cv.SimpleBlobDetector_create(params)

keypoints = detector.detect(img2)

img1 = cv.drawKeypoints(img1, keypoints, np.zeros((1, 1)), (0, 216, 255), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
print(f"Total detected: {len(keypoints)}")

cv.imwrite("./output/output4.jpg", img1)
