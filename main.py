import numpy as np
import cv2 as cv

fp = open("output.txt", "w")

img1 = cv.imread('./input/pollen1_small.jpg')
img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

#   binarize grayscale image
_, img2 = cv.threshold(img1_gray, 105, 255, cv.THRESH_BINARY_INV)
img3 = cv.adaptiveThreshold(img1_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
mask = np.zeros(img1_gray.shape, dtype="uint8")

cv.imwrite("./output/output1.jpg", img2)
cv.imwrite("./output/output2.jpg", img3)

#   dilate to fill the gaps
kernel = np.ones((3, 3), np.uint8)
img2 = cv.dilate(img2, kernel, iterations=3)
img2 = cv.erode(img2, kernel, iterations=3)



canny = cv.Canny(img2, 30, 150, 3)
cnt, heirarchy = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
cv.drawContours(img1, cnt, -1, (0, 255, 0), 2)
print(f"Detected: {len(cnt)}")

for i in cnt:
    fp.write(str(i) + "\n")


cv.imwrite("./output/output3.jpg", img1)
fp.close()