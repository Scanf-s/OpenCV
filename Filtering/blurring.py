import cv2
import cv2 as cv
import numpy as np

def blurring(src, ksize=3):
    dst = cv.blur(src, (ksize, ksize)) # 튜플 형식으로 행렬 크기를 표현해서 넘겨준다.
    description = f"Mean : {ksize} x {ksize}"
    cv2.putText(dst, description, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1, cv2.LINE_AA)
    cv2.imshow("dst", dst)
    cv2.waitKey()

if __name__ == "__main__":
    src = cv.imread("images/img.png", cv.IMREAD_GRAYSCALE)
    if src is None:
        print("Image load failed!")
        exit()

    # mask 크기를 홀수로 지정하는 이유는, 짝수로 만들어버리면 중앙값이 어딘지 명확하지 않기 때문이다.
    for ksize in range(3, 11, 2):
        blurring(src, ksize)

    cv2.destroyAllWindows()