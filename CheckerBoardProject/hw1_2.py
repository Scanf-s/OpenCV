import cv2
import numpy as np
import sys
import argparse


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='이미지 투시 변환 및 밝기 처리 프로그램')
    parser.add_argument('image_path', help='이미지 파일 경로')
    return parser.parse_args()

def perspective_transform(image_path):
    # 이미지 불러오기
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지 파일을 불러올 수 없습니다. : '{image_path}'")
        sys.exit(1)

    # 이미지를 YCrCb로 변환하고 히스토그램 평활화
    ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb_image)
    y = cv2.equalizeHist(y)
    ycrcb_equalized = cv2.merge([y, cr, cb])
    equalized_image = cv2.cvtColor(ycrcb_equalized, cv2.COLOR_YCrCb2BGR)

    points = []

    # 마우스 클릭 이벤트 콜백 함수
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            # 클릭한 위치에 원 그리기
            cv2.circle(equalized_image, (x, y), 2, (0, 0, 255), -1)
            cv2.imshow('Image', equalized_image)

            if len(points) == 4:
                # 선택한 점들을 numpy 배열로 변환
                pts = np.float32(points)

                # 너비와 높이 계산
                width = max(
                    np.linalg.norm(pts[1] - pts[0]),  # 상단 가장자리 길이
                    np.linalg.norm(pts[2] - pts[3])  # 하단 가장자리 길이
                )
                height = max(
                    np.linalg.norm(pts[3] - pts[0]),  # 좌측 가장자리 길이
                    np.linalg.norm(pts[2] - pts[1])  # 우측 가장자리 길이
                )

                # 정사각형 출력을 위해 너비와 높이 중 큰 값을 사용해서 사이즈 결정
                size = max(int(width), int(height))

                # 변환 후 좌표
                dst_pts = np.float32([
                    [0, 0],
                    [size - 1, 0],
                    [size - 1, size - 1],
                    [0, size - 1]
                ])

                # 투시 변환 행렬 계산
                matrix = cv2.getPerspectiveTransform(pts, dst_pts)

                # 투시 변환 적용
                result = cv2.warpPerspective(equalized_image, matrix, (size, size))

                # 밝기 정보를 이용하여 밝은 색은 흰색으로 변환
                hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv)

                # 밝기(V) 채널에서 임계값을 적용하여 마스크 생성
                threshold_value = 160
                _, bright_mask = cv2.threshold(v, threshold_value, 255, cv2.THRESH_BINARY)

                # 밝은 영역을 흰색으로 설정
                result[bright_mask == 255] = [255, 255, 255]

                # HSV를 그레이스케일로 변환 및 이미지 이진화
                result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
                result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                threshold_value = 128
                _, binary_image = cv2.threshold(result, threshold_value, 255, cv2.THRESH_BINARY)

                # 이미지 전처리 (모폴로지 연산 적용)
                # https://wjunsea.tistory.com/5
                # MORPH_OPEN 연산은 침식 연산(객체가 홀쭉해짐) 후, 팽창 연산(객체가 뚱뚱해짐)을 수행한다.
                binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

                cv2.imshow('Result', binary_image)

    print("아래 순서대로 이미지 모서리를 클릭해주세요:")
    print("1. 좌상단 모서리")
    print("2. 우상단 모서리")
    print("3. 우하단 모서리")
    print("4. 좌하단 모서리")

    cv2.imshow('Image', equalized_image)
    cv2.setMouseCallback('Image', mouse_callback)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = parse_arguments()
    perspective_transform(args.image_path)
