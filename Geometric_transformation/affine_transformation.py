import cv2
import numpy as np

def affine_transformation():
    src = cv2.imread('images/img.png', cv2.IMREAD_GRAYSCALE)

    if src is None:
        print('Image load failed!')
        exit()

    rows = src.shape[0] # 이미지 세로 픽셀 수
    cols = src.shape[1] # 이미지 가로 픽셀 수

    # 어파인변환은 설명에서 말했다 싶이, 세개의 점만 알면 됨
    # 원본 이미지의 왼쪽 상단 모서리, 윗변의 맨 오른쪽 좌표, 아랫변의 맨 오른쪽 좌표를 지정
    src_pts = np.array([[0, 0], [cols - 1, 0], [cols - 1, rows - 1]], dtype=np.float32)

    # 새로운 이미지의 왼쪽 상단 모서리를 (50, 50)로 이동시킬것임
    # 오른쪽 상단 모서리를 (cols - 100, 100)으로 이동시킬것임
    # 원본 이미지의 오른쪽 하단 모서리를 (cols - 50, rows - 50)으로 이동시킬것임
    dst_pts = np.array([[50, 50], [cols - 100, 100], [cols - 50, rows - 50]], dtype=np.float32)

    affine_mat = cv2.getAffineTransform(src_pts, dst_pts) # 어파인 변환 행렬을 구하는 함수
    dst = cv2.warpAffine(src, affine_mat, (cols, rows)) # 어파인 변환하는 함수

    cv2.imshow('src', src)
    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    affine_transformation()