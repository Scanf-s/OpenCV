import sys
import os
import cv2

if __name__ == "__main__":
    img = cv2.imread(os.path.join(os.path.dirname(__file__), "images/lenna.bmp"), cv2.IMREAD_GRAYSCALE)
    # imread : 영상 파일을 불러올 때 사용하는 함수
    if img is None:
        print("Failed to read image file")
        sys.exit()

    cv2.imshow("Lenna", img) # 영상을 출력하는 함수
    cv2.waitKey(0)
    cv2.destroyAllWindows()
