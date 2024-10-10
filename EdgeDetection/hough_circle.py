import cv2
import numpy as np

src = cv2.imread("images/img_1.png", cv2.IMREAD_GRAYSCALE)
if src is None:
    exit()

blurred = cv2.blur(src, (3, 3))
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 10, param1=300, param2=50)
dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in range(circles.shape[1]):
        cx, cy, radius = circles[0][i]
        cv2.circle(dst, (cx, cy), radius, (0, 0, 255), 2, cv2.LINE_AA)

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
