import pollen_detector as pd

detector = pd.PollenDetector()

detector.params["hough"]["minRadius"] = 20
detector.detect("./input/pollen1.jpg")
print(f"Detected pollen: {detector.pollenCount}")