import numpy as np
import cv2 as cv

def func3():
    img1 = cv.imread('images/lenna.bmp')  # 이미지 파일을 읽어옴

    img2 = img1                  # 얕은 복사, img1과 메모리를 공유
    img3 = img1.copy()           # 깊은 복사, img1과 별도의 메모리를 사용

    img1[:, :] = (0, 255, 255)   # img1의 모든 픽셀 값을 노란색으로 변경 (BGR 형식)

    # 각각의 이미지를 화면에 출력
    cv.imshow('img1', img1)
    cv.imshow('img2', img2)
    cv.imshow('img3', img3)		 # img1의 원래 정보를 가지고 있음
    cv.waitKey()                 # 키 입력 대기
    cv.destroyAllWindows()       # 모든 창 닫기

def func4():
    img1 = cv.imread('images/lenna.bmp', cv.IMREAD_GRAYSCALE)  # 이미지를 그레이스케일로 읽음

    img2 = img1[200:400, 200:400]         # img1의 200:400 영역을 얕은 복사
    img3 = img1[200:400, 200:400].copy()  # img1의 200:400 영역을 깊은 복사
    img4 = img1[200:400, 200:400].copy()  # img1의 200:400 영역을 깊은 복사

    img2 += 20  # img2의 모든 값에 20을 더함, 얕은 복사라서 img1에도 영향을 줌
    img4 = ~img4 # 영역 반전

    # 각각의 이미지를 화면에 출력
    cv.imshow('img1', img1)
    cv.imshow('img2', img2)
    cv.imshow('img3', img3)		# img3는 기존 이미지를 가지고 있음 (변경 X)
    cv.imshow('img4', img4)
    cv.waitKey()                # 키 입력 대기
    cv.destroyAllWindows()      # 모든 창 닫기

if __name__ == "__main__":
    mat1 = np.zeros((3, 3), np.uint8)
    print(mat1)

    mat2 = np.ones((3, 3), np.uint8)
    print(mat2)

    mat3 = np.eye(3)
    print(mat3)

    mat4 = np.full((480, 640), 0, np.float32) # 크기가 480x640인 행렬, 모든 값이 0.0 (float32 타입)
    print(mat4)

    func3()

    func4()