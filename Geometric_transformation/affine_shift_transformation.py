import cv2
import numpy as np

def affine_shift_transformation():
    src = cv2.imread('images/img.png', cv2.IMREAD_GRAYSCALE)

    if src is None:
        print('Image load failed!')
        exit()

    affine_mat = np.array([[1, 0, 150], [0, 1, 100]], dtype=np.float32) # 왼쪽 하단 방향으로 (150, 100)만큼 평행이동하려고 이렇게 설정함
    dst = cv2.warpAffine(src, affine_mat, (0, 0))

    cv2.imshow('src', src)
    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    affine_shift_transformation()