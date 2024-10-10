import cv2
import numpy as np

src = cv2.imread("images/lenna.bmp", cv2.IMREAD_GRAYSCALE)
if src is None:
    exit()

sobel_dx = cv2.Sobel(src, cv2.CV_32F, 1, 0)
sobel_dy = cv2.Sobel(src, cv2.CV_32F, 0, 1)

size_of_vector = cv2.magnitude(sobel_dx, sobel_dy) # 결과는 실수형 행렬이므로 그레이스케일 형식으로 변환 하여 magnitude에 저장해준다.
magnitude = np.uint8(np.clip(size_of_vector, 0, 255))

# 임계값(threshold)을 적용하여 이진화(binary) 이미지를 생성한다.
# 에지 강도가 180 이상인 픽셀을 255(흰색)로, 그렇지 않은 픽셀을 0(검정색)으로 설정해주었다.
# 임계값은 내맘대로 바꿀 수 있음
_, edge = cv2.threshold(magnitude, 140, 255, cv2.THRESH_BINARY)

cv2.imshow("src", src)
cv2.imshow("sobel_dx", sobel_dx)
cv2.imshow("sobel_dy", sobel_dy)
cv2.imshow("edge", edge)
cv2.waitKey()
cv2.destroyAllWindows()
