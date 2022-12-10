import numpy as np
import cv2 as cv


img1 = cv.imread('./input/pollen2.jpg')
img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)


cv.imwrite("./output/output1.jpg", img1_gray)

#clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#cl_img = clahe.apply(img1_gray)

#cv.imwrite("./output/output2.jpg", cl_img)

_, img2 = cv.threshold(img1_gray, 60, 255, cv.THRESH_BINARY)
#_, img3 = cv.threshold(cl_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)


img3 = cv.erode(img2, np.ones((3, 3), np.uint8), iterations=10)
img3 = cv.dilate(img2, np.ones((3, 3), np.uint8), iterations=10)
cv.imwrite("./output/output3.jpg", img2)


contours, _ = cv.findContours(~img3, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

cv.drawContours(img1, contours, -1, (0, 255, 0), 3)
print(len(contours))
    


cv.imwrite("./output/output4.jpg", img1)
