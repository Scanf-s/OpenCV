import cv2

def affine_rotation():
    src = cv2.imread("images/img.png", cv2.IMREAD_GRAYSCALE)

    if src is None:
        print("Image load failed!")
        exit()

    # center에 영상 중심을 기준으로 지정
    # 시계 방향으로 45도 반시계 방향으로 회전시킬것임
    affine_mat = cv2.getRotationMatrix2D((src.shape[1] / 2, src.shape[0] / 2), 45, 1)

    dst = cv2.warpAffine(src, affine_mat, (0, 0))

    cv2.imshow("src", src)
    cv2.imshow("dst", dst)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    affine_rotation()