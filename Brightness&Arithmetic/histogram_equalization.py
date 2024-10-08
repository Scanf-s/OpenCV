import cv2
import numpy as np

def calcGrayHist(src=None):
    if src is None:  # 이미지가 로드되지 않은 경우
        print('Image load failed!')
        exit()

    # OpenCV의 calcHist 함수를 사용하여 히스토그램 계산
    channels = [0]  # 이미지의 채널 선택 (그레이스케일이므로 채널은 0)
    histSize = [256]  # 히스토그램의 bin 크기 (픽셀 값의 범위가 0~255 이므로 256개)
    histRange = [0, 256]  # 픽셀 값의 범위 (0부터 256까지)

    hist = cv2.calcHist([src], channels, None, histSize, histRange)
    # None은 마스크를 의미하며, 여기서는 사용하지 않음

    return hist  # 계산된 히스토그램 반환

# 계산된 히스토그램을 이미지로 변환하여 시각화하는 함수
def getGrayHistImage(hist):
    _, histMax, _, _ = cv2.minMaxLoc(hist)  # 히스토그램에서 가장 큰 값을 찾아 히스토그램의 크기를 결정
    # histMax는 히스토그램 막대 그래프의 최대 높이를 설정하기 위한 기준값

    imgHist = np.ones((100, 256), dtype=np.uint8) * 255  # 히스토그램 이미지를 생성, 크기: 100x256, 흰색(255)으로 채움
    for x in range(imgHist.shape[1]):  # 히스토그램의 각 bin에 대해 그래프를 그리기 위한 반복문
        pt1 = (x, 100)  # 막대 그래프의 아래쪽 좌표
        pt2 = (x, 100 - int(hist[x, 0] * 100 / histMax))  # 막대 그래프의 위쪽 좌표 (히스토그램 값에 따라 조정)
        cv2.line(imgHist, pt1, pt2, 0)  # 막대를 그리기 위해 cv2.line 사용 (검정색)

    return imgHist  # 그려진 히스토그램 이미지 반환

def histogram_equalization():
    src = cv2.imread('images/img.png', cv2.IMREAD_GRAYSCALE)

    if src is None:
        print('Image load failed!')
        exit()

    dst = cv2.equalizeHist(src)

    cv2.imshow('src', src)
    cv2.imshow('srcHist', getGrayHistImage(calcGrayHist(src)))

    cv2.imshow('dst', dst)
    cv2.imshow('dstHist', getGrayHistImage(calcGrayHist(dst)))
    cv2.waitKey()
    cv2.destroyAllWindows()

histogram_equalization()
