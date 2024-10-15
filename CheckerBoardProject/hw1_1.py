import cv2
import numpy as np

WINDOW_SIZE: tuple[int, int] = (800, 800)
BLUR_SIZE: tuple[int, int] = (3, 3)

def resize_image(image: cv2.Mat, min_size: int = 500, max_size: int = 1000) -> cv2.Mat:
    height, width = image.shape[:2]
    scale = 1.0

    # 이미지가 너무 작은 경우 확대
    if min(height, width) < min_size:
        scale = min_size / min(height, width)
    # 이미지가 너무 큰 경우 축소
    elif max(height, width) > max_size:
        scale = max_size / max(height, width)

    if scale != 1.0:
        resized_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return resized_image
    return image  # 크기 조정이 불필요한 경우 원본 이미지 반환

def show_image(window_name: str, image: cv2.Mat, window_size: tuple = WINDOW_SIZE) -> None:
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, *window_size)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)

def edge_detection(image: cv2.Mat) -> cv2.Mat:
    # Noise 제거
    blurred: cv2.Mat = cv2.GaussianBlur(image, BLUR_SIZE, 0)

    # Median 임계값 계산
    median: float = float(np.median(blurred))
    low_threshold = int(max(0, (1.0 - 0.33) * median))
    high_threshold = int(min(255, (1.0 + 0.33) * median))

    # Canny Edge 검출
    canny: cv2.Mat = cv2.Canny(blurred, low_threshold, high_threshold)
    return canny

def detect_checkerboard_size(edge_image: cv2.Mat, original_image: cv2.Mat) -> tuple[int, int]:
    # 외곽선 검출
    contours, _ = cv2.findContours(edge_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)

    # 코너 포인트 저장할 리스트
    corner_points = []

    for cnt in contours:
        # 근사화 정확도 조절
        epsilon = 0.013 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        # 다각형이 사각형일 경우에만 처리
        if len(approx) == 4 :
            for point in approx:
                corner = point[0]
                corner_points.append(corner)

    if not corner_points:
        print("코너를 찾을 수 없습니다.")
        return (0, 0)

    # 중복된 코너 제거
    corner_points = np.unique(corner_points, axis=0)

    # 코너의 x, y 좌표 추출
    x_coords = corner_points[:, 0]
    y_coords = corner_points[:, 1]

    # x 좌표 클러스터링
    x_sorted = np.sort(x_coords)
    x_diffs = np.diff(x_sorted)
    x_median_diff = np.median(x_diffs)
    x_threshold = x_median_diff * 0.5 if x_median_diff > 0 else 10  # 임계값 설정

    x_clusters = [x_sorted[0]]
    for x in x_sorted[1:]:
        if abs(x - x_clusters[-1]) > x_threshold:
            x_clusters.append(x)
    num_x_clusters = len(x_clusters)

    # y 좌표 클러스터링
    y_sorted = np.sort(y_coords)
    y_diffs = np.diff(y_sorted)
    y_median_diff = np.median(y_diffs)
    y_threshold = y_median_diff * 0.5 if y_median_diff > 0 else 10  # 임계값 설정

    y_clusters = [y_sorted[0]]
    for y in y_sorted[1:]:
        if abs(y - y_clusters[-1]) > y_threshold:
            y_clusters.append(y)
    num_y_clusters = len(y_clusters)

    # 결과 출력
    print(f'가로 방향 코너 수: {num_x_clusters}')
    print(f'세로 방향 코너 수: {num_y_clusters}')

    # 체커보드의 칸 수 계산
    num_squares_x = num_x_clusters - 1
    num_squares_y = num_y_clusters - 1

    print(f'체커보드 크기: {num_squares_x} x {num_squares_y}')

    # 코너 포인트를 원본 이미지에 표시 (선택 사항)
    for point in corner_points:
        cv2.circle(original_image, tuple(point), 5, (0, 0, 255), -1)
    show_image("Corners Detected", original_image)

    return num_squares_x, num_squares_y

if __name__ == "__main__":
    # 이미지 Grayscale로 열기
    image_path: str = "images/img.png"
    image: cv2.Mat = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("해당 경로에서 이미지를 찾을 수 없습니다. 이미지 경로를 확인하세요.")
        exit()

    # 이미지 크기 조정 및 출력
    image = resize_image(image)
    show_image(window_name="Source", image=image)

    # 히스토그램 평활화
    equalized = cv2.equalizeHist(image)

    # 정확한 검출을 위한 이미지 이진화
    binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]
    show_image(window_name="Binary", image=binary)

    # Canny Edge detecting 수행
    edge_detected_image: cv2.Mat = edge_detection(image)
    show_image(window_name="Canny", image=edge_detected_image)

    # 사각형 검출
    num_squares_x, num_squares_y = detect_checkerboard_size(edge_detected_image, image)

    cv2.destroyAllWindows()
