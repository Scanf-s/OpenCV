import cv2
import numpy as np

src = cv2.imread('images/building_2.png', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('Image load failed!')
    exit()

# Canny edge 검출기
edge = cv2.Canny(src, 50, 150)

# HoughLines 함수로 직선 검출
lines = cv2.HoughLines(edge, 1, np.pi / 180, 130) # 맨 뒤 파라미터 임계값을 조절해서 직선 판정을 늘릴 수 있다.
dst = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

if lines is not None:
    for i in range(lines.shape[0]):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        cos = np.cos(theta)
        sin = np.sin(theta)
        x0 = cos * rho
        y0 = sin * rho
        alpha = 1000  # 직선의 길이를 적절히 조정

        # 두 점을 계산하여 직선을 그림
        pt1 = (int(x0 - alpha * sin), int(y0 + alpha * cos))
        pt2 = (int(x0 + alpha * sin), int(y0 - alpha * cos))
        cv2.line(dst, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)

# 결과 시각화
cv2.imshow('src', src)
cv2.imshow('edge', edge)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
