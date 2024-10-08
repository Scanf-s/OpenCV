import cv2 as cv
import numpy as np

def blurring(src, ksize=3):
    dst = cv


if __name__ : "__main__":
    src = cv.imread("images/img.png", cv.IMREAD_GRAYSCALE)
    if src is None:
        print("Image load failed!")
        exit()

    # mask 크기를 홀수로 지정하는 이유는, 짝수로 만들어버리면 중앙값이 어딘지 명확하지 않기 때문이다.
    for ksize in range(3, 9, 2):
        blurring(src, ksize)