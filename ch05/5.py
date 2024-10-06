import cv2 as cv
import numpy as np

def calcGrayHist(src = None):
    if src is None:
        print('Image load failed!')
        exit()

    # OpenCV's calcHist
    channels = [0] # 여러개의 channel을 지원한다.
    histSize = [256] # bin 크기 지정
    histRange = [0, 256] # 범위

    hist = cv.calcHist([src], channels, None, histSize, histRange)
    # 중간의 None은 마스크이다.

    return hist

def getGrayHistImage(hist):
    _, histMax, _, _ = cv.minMaxLoc(hist)

    imgHist = np.ones((100, 256), dtype=np.uint8) * 255
    for x in range(imgHist.shape[1]):
        pt1 = (x, 100)
        pt2 = (x, 100 - int(hist[x, 0] * 100 / histMax))
        cv.line(imgHist, pt1, pt2, 0)

    return imgHist


src = cv.imread('images/img.png', cv.IMREAD_GRAYSCALE)

hist = calcGrayHist(src)
imgHist = getGrayHistImage(hist)

cv.imshow('src', src)
cv.imshow('imgHist', imgHist)
cv.waitKey()
cv.destroyAllWindows()