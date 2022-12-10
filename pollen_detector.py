import numpy as np
import cv2 as cv

class PollenDetector:
    # Stores default parameters for image processing methods
    params = {
        "blurKernel": (7, 7),
        "canny": (30, 35),
        "hough": {
            "method": cv.HOUGH_GRADIENT,
            "dp": 1,                        # inverse ratio resolution
            "minDist": 35,                  # minimum distance between circle centers
            "param1": 30,                   # upper threshold for Canny
            "param2": 20,                   # threshold for center detection; lower = more circles but prone to false circles
            "minRadius": 20,                # minimum circle radius
            "maxRadius": 55                 # max radius
        },
        "threshold": {
            "thresh": 60,                   # threshold value for binarization
            "maxval": 255,                  # maximum value for THRESH_BINARY
            "type": cv.THRESH_BINARY        # thresholding type
        },
        "binaryop": {
            "kernel": np.ones((3, 3), np.uint8),    # structuring element used for dilation/erosion
            "iterations": 10                        # number of times dilation/erosion is applied
        }
    }

    def __init__(self):
        self.img = None
        self.pollenCount = 0
        self.darkPollen = 0
        self.lightPollen = 0
    
    # [PRIVATE]: performs hough circle transform on given image 
    def __hough(self, src):
        def __preprocessing(src):
            output = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
            output = cv.GaussianBlur(output, self.params["blurKernel"], 0)
            #output = cv.Canny(output, self.params["canny"][0], self.params["canny"][1])

            return output

        src = __preprocessing(src)

        return cv.HoughCircles(
            src,
            self.params["hough"]["method"],
            self.params["hough"]["dp"],
            self.params["hough"]["minDist"],
            param1=self.params["hough"]["param1"],
            param2=self.params["hough"]["param2"],
            minRadius=self.params["hough"]["minRadius"],
            maxRadius=self.params["hough"]["maxRadius"]
        )
    
    # [PRIVATE]: detects contours on a given image
    def __contour(self, src):
        def __preprocessing(src):
            output = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

            # binarization
            _, output = cv.threshold(
                output, 
                self.params["threshold"]["thresh"], 
                self.params["threshold"]["maxval"], 
                self.params["threshold"]["type"]
            )

            # binary opening
            output2 = cv.erode(output, 
                self.params["binaryop"]["kernel"],
                iterations=self.params["binaryop"]["iterations"]
            )
            output = cv.dilate(output, 
                self.params["binaryop"]["kernel"],
                iterations=self.params["binaryop"]["iterations"]
            )

            return output
        
        src = __preprocessing(src)

        return cv.findContours(
            ~src,
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE
        )

    # detects pollen from a given image; returns a tuple of hough circles and contours
    def detect(self, filepath):
        self.img = cv.imread(filepath)

        # detect all pollen
        circles = self.__hough(self.img)
        circles = np.uint16(np.around(circles))
        self.pollenCount = len(circles[0,:])

        # detect dark pollen
        contours, _ = self.__contour(self.img)
        self.darkPollen = len(contours)
        self.lightPollen = self.pollenCount - self.darkPollen

        return (circles, contours)
    
    def drawHoughCircles(self, filepath, circles):
        self.img = cv.imread(filepath)
        for i in circles[0,:]:
            drawn = cv.circle(self.img, (i[0],i[1]), i[2], (0, 255, 0), 4)
            drawn = cv.circle(self.img, (i[0],i[1]), 2, (0, 0, 255), 2)

        cv.imwrite('./output/_allpollen.jpg', drawn)

        return drawn

    def drawContours(self, filepath, contours):
        self.img = cv.imread(filepath)
        drawn = cv.drawContours(self.img, contours, -1, (0, 255, 0), 3)

        cv.imwrite('./output/_darkpollen.jpg', drawn)

        return drawn