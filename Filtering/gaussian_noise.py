import cv2
import numpy as np

src = cv2.imread("images/img.png", cv2.IMREAD_GRAYSCALE)
if src is None:
    print("Image load failed!")
    exit()

cv2.imshow("src", src)

for standard_deviation in [10, 20, 30]:
    noise = np.zeros(src.shape, np.int32)
    cv2.randn(noise, 0, standard_deviation) # standard deviation (표준편차를 사용해서 가우시안 노이즈 행렬 생성)

    dst = cv2.add(src, noise, dtype=cv2.CV_8UC1)

    cv2.putText(dst, f"sigma = {standard_deviation}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1, cv2.LINE_AA)

    cv2.imshow("dst", dst)
    cv2.waitKey()

cv2.destroyAllWindows()