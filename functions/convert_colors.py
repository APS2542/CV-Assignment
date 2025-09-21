import cv2 as cv

#convert image color (RGB > Grey)
def convertgray(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#convert image color (RGB > HSV)
def converthsv(img):
    return cv.cvtColor(img, cv.COLOR_BGR2HSV)

def demo(img):
    gray = convertgray(img)
    hsv = converthsv(img)

    gray_bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    hsv_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    combined = cv.hconcat([img, gray_bgr, hsv_bgr])
    return combined