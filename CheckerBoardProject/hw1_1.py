import cv2
import numpy as np
import sys
import argparse

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='체커보드 크기 검출 프로그램')
    parser.add_argument('image_path', help='체커보드 이미지 경로')
    return parser.parse_args()

def resize_image(image: cv2.Mat, min_size: int = 500, max_size: int = 1000) -> cv2.Mat:
    # 입력 이미지 크기 조절
    # height, width 중 더 큰 값을 기준으로 정방형 행렬로 조절해주는 함수
    height, width = image.shape[:2]
    scale = 1.0

    if min(height, width) < min_size:
        scale = min_size / min(height, width)
    elif max(height, width) > max_size:
        scale = max_size / max(height, width)

    if scale != 1.0:
        resized_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return resized_image
    return image

def preprocess_image(image: cv2.Mat) -> cv2.Mat:
    # 노이즈 제거를 위한 블러링
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    # 대비 조절 (히스토그램 평활화)
    equalized = cv2.equalizeHist(blurred)
    # Canny 엣지 검출
    edges = cv2.Canny(equalized, 50, 150, apertureSize=3)
    return edges

def detect_lines(edges: cv2.Mat) -> tuple[np.ndarray|None, np.ndarray|None]:
    # 허프 변환을 이용하여 직선 검출
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
    if lines is None:
        print("직선을 찾을 수 없습니다.")
        return None, None

    # 수평선과 수직선 분류
    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        rho, theta = line[0]
        angle = theta * (180 / np.pi)
        if (angle < 10 or angle > 170):
            # 수직선
            vertical_lines.append((rho, theta))
        elif (80 < angle < 100):
            # 수평선
            horizontal_lines.append((rho, theta))

    if len(horizontal_lines) == 0 or len(vertical_lines) == 0:
        print("수평선 또는 수직선을 충분히 찾을 수 없습니다.")
        return None, None

    return horizontal_lines, vertical_lines

def compute_line_points(lines: np.ndarray, img_shape: tuple) -> list:
    # 이미지의 크기를 이용하여 직선의 시작점과 끝점을 계산
    line_points = []
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # 직선의 시작점과 끝점 계산
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        line_points.append(((x1, y1), (x2, y2)))
    return line_points

def find_intersections(h_lines: list, v_lines: list) -> np.ndarray:
    # 수평선과 수직선의 교차점 계산
    intersections = []
    for h_line in h_lines:
        for v_line in v_lines:
            pt = compute_intersection(h_line, v_line)
            if pt is not None:
                intersections.append(pt)
    return np.array(intersections)

def compute_intersection(line1: tuple, line2: tuple) -> tuple | None:
    # 두 직선의 교차점을 계산
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2

    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denominator == 0:
        return None  # 평행선

    px = ((x1 * y2 - y1 * x2)*(x3 - x4) - (x1 - x2)*(x3 * y4 - y3 * x4)) / denominator
    py = ((x1 * y2 - y1 * x2)*(y3 - y4) - (y1 - y2)*(x3 * y4 - y3 * x4)) / denominator

    return int(px), int(py)

def cluster_points(points: np.ndarray, distance_threshold: int = 20) -> np.ndarray:
    # points를 클러스터링하여 중복 제거
    clustered_points = []
    for point in points:
        if not any(np.linalg.norm(point - np.array(other_point)) < distance_threshold for other_point in clustered_points):
            clustered_points.append(point)
    return np.array(clustered_points)

def sort_grid_points(points: np.ndarray) -> tuple[int, int]:
    # points를 그리드 형태로 정렬하고 행과 열의 개수를 계산
    # y 좌표 기준으로 정렬하여 행 분류
    sorted_points = points[points[:,1].argsort()]
    rows = []
    current_row = [sorted_points[0]]
    for point in sorted_points[1:]:
        if abs(point[1] - current_row[-1][1]) < 20:
            current_row.append(point)
        else:
            rows.append(current_row)
            current_row = [point]
    rows.append(current_row)

    # 각 행에서 x 좌표로 정렬
    for row in rows:
        row.sort(key=lambda p: p[0])

    num_rows = len(rows)
    num_cols = max(len(row) for row in rows)

    return num_cols - 1, num_rows - 1  # 내부 교차점 개수이므로 -1

def visualize_result(image: cv2.Mat, points: np.ndarray, h_lines_points: list, v_lines_points: list):
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # 교차점 표시
    for point in points:
        cv2.circle(image_color, tuple(point), 5, (0, 0, 255), -1)
    # 선분 표시
    for line in h_lines_points + v_lines_points:
        cv2.line(image_color, line[0], line[1], (0, 255, 0), 1)
    cv2.imshow('Result', image_color)
    cv2.waitKey(0)

def main():
    args = parse_arguments()
    image: cv2.Mat = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"이미지 파일을 불러올 수 없습니다. : '{args.image_path}'")
        sys.exit(1)

    image = resize_image(image)

    edges = preprocess_image(image)

    h_lines, v_lines = detect_lines(edges)
    if h_lines is None or v_lines is None:
        sys.exit(1)

    # 직선의 시작점과 끝점 계산
    h_lines_points = compute_line_points(h_lines, image.shape)
    v_lines_points = compute_line_points(v_lines, image.shape)

    # 교차점 계산
    intersections = find_intersections(h_lines_points, v_lines_points)
    if len(intersections) == 0:
        print("교차점을 찾을 수 없습니다.")
        sys.exit(1)

    # 교차점 클러스터링 (분류)
    clustered_points = cluster_points(intersections)
    if len(clustered_points) == 0:
        print("코너로 선택된 점들을 클러스터링할 수 없습니다.")
        sys.exit(1)

    # 칸 수 계산
    num_squares_x, num_squares_y = sort_grid_points(clustered_points)

    # 클러스터링 된 포인트가 이미지 잡음에 의해 오류가 발생하는 경우가 존재해서
    # num_squares_x와 num_squares_y 중 더 작은 값을 사용해주었다.
    result = min(num_squares_x, num_squares_y)

    print(f"체커보드 크기: {result} x {result}")

    # 결과 시각화
    visualize_result(image, clustered_points, h_lines_points, v_lines_points)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
