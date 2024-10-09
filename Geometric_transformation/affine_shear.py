import cv2
import numpy as np

def affine_shear():
    src = cv2.imread("images/img.png", cv2.IMREAD_GRAYSCALE)

    if src is None:
        print("Image load failed!")
        exit()

    rows = src.shape[0]
    cols = src.shape[1]

    mx = 0.3 # 얼만큼 x좌표를 밀어버릴건지
    affine_mat = np.array([[1, mx, 0], [0, 1, 0]], dtype=np.float32)

    # 맨 마지막 파라미터는 data size 파라미터인데,
    # dst 영상의 가로 크기는 x좌표가 밀렸으니까 src.cols + src.rows * mx로 설정해야함
    # dst 영상의 세로 크기는 안밀렸으니까 그대로 row로 설정
    dst = cv2.warpAffine(src, affine_mat, (int(cols + rows * mx), rows))

    cv2.imshow("src", src)
    cv2.imshow("dst", dst)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    affine_shear()