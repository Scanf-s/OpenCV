import cv2

def affine_resize():
    src = cv2.imread('images/img.png', cv2.IMREAD_GRAYSCALE)

    if src is None:
        print('Image load failed!')
        exit()

    # 가로세로 4배씩 늘림
    dst = cv2.resize(src, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST)

    # 가로로 2배, 세로로 2배 축소
    dst2 = cv2.resize(src, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    cv2.imshow('src', src)
    cv2.imshow('dst', dst[400:800, 500:900]) # 너무 커져서 일부만 표시
    cv2.imshow('dst2', dst2)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    affine_resize()