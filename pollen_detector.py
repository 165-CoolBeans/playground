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
        }
    }

    def __init__(self):
        self.img = None
        self.pollenCount = 0
    
    # [PRIVATE]: performs hough circe transform on given image 
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

    def detect(self, src):
        self.img = cv.imread(src)

        circles = self.__hough(self.img)
        circles = np.uint16(np.around(circles))
        self.pollenCount = len(circles[0,:])

        return circles