import cv2
import numpy as np
import sys
import argparse

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='체커보드 말 검출 프로그램')
    parser.add_argument('image_path', help='체커보드 이미지 파일 경로')
    return parser.parse_args()

def count_checker_pieces(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: 이미지 파일 '{image_path}'을(를) 읽을 수 없습니다.")
        sys.exit(1)

    cv2.imshow('image', image)

    # 이미지를 그레이스케일로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 가우시안 블러 적용
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # 샤프닝 커널 정의
    sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # 이미지 샤프닝 적용
    sharpened = cv2.filter2D(blurred, -1, sharpening_kernel)

    # 적응형 이진화 적용
    adaptive_thresh = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # 가우시안 분포에 따른 가중치 합으로 임계값을 결정
        cv2.THRESH_BINARY,
        9, # 임계값을 적용할 영역의 사이즈
        2
    )

    # 엣지 검출
    edges = cv2.Canny(adaptive_thresh, 50, 150)

    # 허프 변환 원 검출
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=0.95,
        minDist=20,
        param1=50,
        param2=25,
        minRadius=5,
        maxRadius=18
    )

    if circles is None:
        print("말을 찾을 수 없습니다.")
        sys.exit(1)

    circles = np.uint16(np.around(circles))

    # 밝은 색 말과 어두운 색 말의 수를 저장할 변수
    light_count = 0  # 밝은 색 (흰색에 가까운 말)
    dark_count = 0   # 어두운 색 (검정색에 가까운 말)

    # 결과 이미지를 위한 복사본 생성
    output = image.copy()

    for i in circles[0, :]:
        # 원의 중심 좌표와 반지름 가져오기
        center = (i[0], i[1])
        radius = i[2]

        # 중심점의 색상 가져오기
        center_color = image[center[1], center[0]]

        # BGR을 그레이스케일 밝기로 변환
        center_intensity = 0.299 * center_color[2] + 0.587 * center_color[1] + 0.114 * center_color[0]

        # 밝기 값을 기준으로 말의 색상 분류
        intensity_threshold = 130
        if center_intensity >= intensity_threshold:
            color = (0, 255, 0)  # 녹색 원 (밝은 색 말)
            light_count += 1
        else:
            color = (0, 0, 255)  # 빨간색 원 (어두운 색 말)
            dark_count += 1

        # 원의 경계 그리기
        cv2.circle(output, center, radius, color, 2)
        # 중심점 그리기
        cv2.circle(output, center, 2, color, 2)

    print(f"w: {light_count} b: {dark_count}")
    cv2.imshow('Detected Pieces', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    args = parse_arguments()

    count_checker_pieces(args.image_path)

if __name__ == "__main__":
    main()
