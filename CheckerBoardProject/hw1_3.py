import cv2
import numpy as np
import sys
import argparse

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='체커보드 이미지 자동 투시 변환 프로그램')
    parser.add_argument('image_path', help='체커보드 이미지 파일 경로')
    return parser.parse_args()

def find_checkerboard_corners(image):
    # 이미지를 그레이스케일로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 엣지 검출을 위해 Gaussian Blur와 Canny Edge Detection 적용
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # 외곽선 검출
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 가장 큰 사각형 외곽선을 찾기
    largest_contour = None
    max_area = 0
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):  # 사각형이며 볼록한 외곽선일 경우
            area = cv2.contourArea(approx)
            if area > max_area:
                largest_contour = approx
                max_area = area

    if largest_contour is None:
        print("체커보드 코너를 찾을 수 없습니다.")
        sys.exit(1)

    # 좌표 정렬 (좌상단, 우상단, 우하단, 좌하단 순으로)
    corners = largest_contour[:, 0, :]

    # 좌표를 정렬하여 순서 지정
    sum_coords = corners.sum(axis=1)
    diff_coords = np.diff(corners, axis=1)

    top_left = corners[np.argmin(sum_coords)]
    bottom_right = corners[np.argmax(sum_coords)]
    top_right = corners[np.argmin(diff_coords)]
    bottom_left = corners[np.argmax(diff_coords)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

def perspective_transform(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지 파일을 불러올 수 없습니다. : '{image_path}'")
        sys.exit(1)
    cv2.imshow('original', image)

    # 체커보드 코너 찾기
    corners = find_checkerboard_corners(image)

    # 코너 순서대로 정렬 (좌상단, 우상단, 우하단, 좌하단)
    width, height = 500, 500
    dst_pts = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])

    # 투시 변환 행렬 계산 및 적용
    matrix = cv2.getPerspectiveTransform(corners, dst_pts)
    result = cv2.warpPerspective(image, matrix, (width, height))

    return result

if __name__ == "__main__":
    args = parse_arguments()
    result = perspective_transform(args.image_path)

    # 결과 이미지 표시
    cv2.imshow('Transformed', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
