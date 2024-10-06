import cv2 as cv

def contrast():
    src = cv.imread("images/img.png", cv.IMREAD_GRAYSCALE)
    if src is None:
        print("Image load failed!")
        exit()

    src2 = src.copy()

    s1 = 2.0 # 전체적으로 명암비를 높인다.
    s2 = 0.5 # 전체적으로 명암비를 낮춘다.

    dst1 = cv.multiply(src, s1)
    dst2 = cv.multiply(src2, s2)

    cv.imshow("src", src) # 원본 이미지
    cv.imshow("dst1", dst1) # 명암비를 높인 이미지
    cv.imshow("dst2", dst2) # 명암비를 낮춘 이미지
    cv.waitKey()
    cv.destroyAllWindows()

contrast()