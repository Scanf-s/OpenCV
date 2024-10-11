import cv2
import os
import numpy as np

# 이미지 파일 경로 확인
image_path = "/home/sullung/Dev/OpenCV/EdgeDetection/images/img.png"
if not os.path.exists(image_path):
    print("이미지 파일을 찾을 수 없습니다.")
    exit()

src = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if src is None:
    print("이미지를 불러오는 데 실패했습니다.")
    exit()

# 이미지 크기 조정
max_size = 800
height, width = src.shape[:2]
if max(height, width) > max_size:
    scale = max_size / max(height, width)
    src = cv2.resize(src, None, fx=scale, fy=scale)

# 가우시안 블러 적용
blurred = cv2.GaussianBlur(src, (5, 5), 0)

# 자동 임계값 계산
median = np.median(blurred)
lower = int(max(0, (1.0 - 0.33) * median))
upper = int(min(255, (1.0 + 0.33) * median))

# Canny 엣지 검출 적용
canny = cv2.Canny(blurred, lower, upper)

cv2.imshow("src", src)
cv2.imshow("canny", canny)
cv2.waitKey()
cv2.destroyAllWindows()