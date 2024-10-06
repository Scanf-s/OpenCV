import cv2 as cv

def contrast():
    src = cv.imread('images/img.png', cv.IMREAD_GRAYSCALE)

    if src is None:
        print('Image load failed!')
        exit()

    # 명암비 조절을 위한 알파 값 (1.0을 기준으로 명암비를 변경할 수 있음)
    alpha = 0.5 # 명암비 조절 강

    # convertScaleAbs를 사용하여 이미지의 명암비를 조절
    dst = cv.convertScaleAbs(src=src, alpha=alpha + 1, beta=-128 * alpha)

    cv.imshow('src', src)
    cv.imshow('dst', dst)
    cv.waitKey()
    cv.destroyAllWindows()

contrast()