import cv2
import numpy as np

src = cv2.imread("images/img.png", cv2.IMREAD_GRAYSCALE)
if src is None:
    print("Image load failed!")
    exit()

# 1. 먼저 노이즈가 있는 영상 생성
noise = np.zeros(src.shape, np.int32)
cv2.randn(noise, 0, 5) # 평균이 0이고 표준편차가 5인 가우시안 노이즈 행렬 생성
cv2.add(src, noise, src, dtype=cv2.CV_8UC1) # 노이즈 적용

dst1 = cv2.GaussianBlur(src, (0, 0), 5)
# sigmaSpace : 일반적인 가우시안 필터링에서 사용하는 표준편차 값 -> 값이 클수록 더 많은 주변 픽셀을 고려해서 블러링
# sigmaColor : 주변 픽셀과의 밝기 차이에 관한 표준편차 값 -> 값이 작을수록 픽셀 값 차이가 큰 주변 픽셀은 블러링 적용 X
dst2 = cv2.bilateralFilter(src, -1, sigmaColor=10, sigmaSpace=5)

cv2.imshow("src", src)
cv2.imshow("dst", dst1)
cv2.imshow("dst2", dst2)
cv2.waitKey()
cv2.destroyAllWindows()