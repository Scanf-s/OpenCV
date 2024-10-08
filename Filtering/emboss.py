import cv2 as cv
import numpy as np

src = cv.imread("images/img.png", cv.IMREAD_GRAYSCALE)
if src is None:
    print("Image load failed!")
    exit()

emboss_mask = np.array(
    [[1, 1, 0],
     [1, 0, -1],
     [0, -1, -1]],
    np.float32
)

dst = cv.filter2D(src, -1, emboss_mask, delta=128)

cv.imshow('src', src)
cv.imshow('dst', dst)
cv.waitKey()
cv.destroyAllWindows()