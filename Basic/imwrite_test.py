import numpy as np
import cv2 as cv
import os
import sys

if __name__ == "__main__":
    img = cv.imread(os.path.join(os.path.dirname(__file__), "images/lenna.bmp"), cv.IMREAD_GRAYSCALE)
    if img is None:
        print("Failed to read image file")
        sys.exit()

    print(f"1. Type of img: {type(img)}") # img의 타입 -> python은 numpy의 ndarray로 처리
    print(f"2. Shape of img: {img.shape}") # img의 모양 -> (512, 512)

    if len(img.shape) == 2:
        print("Grayscale image")
    elif len(img.shape) == 3:
        print("Color image")
    else:
        print("Unknown image type")

    cv.imshow("Lenna", img)
    cv.waitKey()
    cv.destroyAllWindows()
