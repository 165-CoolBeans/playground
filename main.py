import pollen_detector as pd

detector = pd.PollenDetector()
currentFile = "./input/pollen2.jpg"

detector.params["hough"]["minRadius"] = 20
hcircles, contours = detector.detect(currentFile)
print(f"""
    Detected pollen: {detector.pollenCount}
    Dark pollen: {detector.darkPollen}
    Light pollen: {detector.lightPollen}
""")

detector.drawContours(currentFile, contours)
detector.drawHoughCircles(currentFile, hcircles)