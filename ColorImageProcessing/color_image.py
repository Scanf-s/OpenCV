import cv2
import numpy as np


src = cv2.imread("images/img.png", cv2.IMREAD_COLOR)
if src is None:
    exit()

print(src.shape) # 이미지의 (세로, 가로, 색상 채널)
print(src.dtype) # 이미지 타입 -> Python에서는 uint8 사용
print(src[0, 0]) # 픽셀 좌표(0, 0)의 (B, G, R)값

cv2.imshow("src", src)
cv2.waitKey()

# inverse
src = ~src
cv2.imshow("src", src)
cv2.waitKey()

# 비효율적인 inverse 연산
dst = np.zeros(src.shape, src.dtype)
for y in range(src.shape[0]):
    for x in range(src.shape[1]):
        dst[y, x] = 255 - src[y, x] # 255에서 src[y, x] BGR값을 각각 빼는 연산

cv2.imshow("dst", dst)
cv2.waitKey()

cv2.destroyAllWindows()
