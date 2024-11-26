import cv2
import numpy as np

src = cv2.imread("images/img.png", cv2.IMREAD_GRAYSCALE)
if src is None:
    exit()

arr = np.array([[1, 2, 3], [4, 5, 6]])

print(arr[:, :]) #전체 출력
print(arr[0, :]) #첫번째 행만 출력
print(arr[:, 1]) #2번째 열만 출력
print(arr[1, 1:]) #두번째 행의 2,3번째 열 요소 출럭


arr2 = np.array(3)
print(arr2)
print(arr2.ndim) # 상수는 0차원임
print(arr2.shape) # 0차원이니까 행과 열 정보 X

# 영상을 한번에 반전하고 싶을 때
# src2 = src.copy()
# src2 = ~src2
#
# # 영상의 오른쪽 절반을 검정색으로 바꾸고 싶다면?
# src3 = src.copy()
# src3[:, src3.shape[1]//2:] = 0 # 모든 행에 대해 적용하는데, 열 기준으로 오른쪽 부분만 값을 0으로 설정
#
# cv2.imshow("src", src)
# cv2.imshow("src2", src2)
# cv2.imshow("src3", src3)
#
# cv2.waitKey()
# cv2.destroyAllWindows()

