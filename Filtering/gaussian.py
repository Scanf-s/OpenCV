import cv2 as cv
import numpy as np

def gaussian(src, sigma):
    dst = cv.GaussianBlur(src, (0, 0), sigma)
    cv.putText(dst, f"sigma = {sigma}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1, cv.LINE_AA)
    cv.imshow("dst", dst)
    cv.waitKey()

if __name__ == "__main__":
    src = cv.imread("images/img.png", cv.IMREAD_GRAYSCALE)
    if src is None:
        print("Image load failed!")
        exit()

    cv.imshow("src", src)

    for sigma in range(1, 6):
        gaussian(src, sigma)

    cv.destroyAllWindows()
