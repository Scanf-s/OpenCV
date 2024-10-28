import cv2
import numpy as np
import sys
from typing import List, Tuple


class CheckerboardPerspectiveTransformer:
    def __init__(self, image_path: str, output_size: int = 800):
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

        # 초기 이미지 크기 조정
        self.image = self.resize_image(self.original_image.copy())
        self.corner_points: List[Tuple[int, int]] = []
        self.output_size = output_size
        self.scale_factor = self.image.shape[0] / self.original_image.shape[0]

    def resize_image(self, image: np.ndarray, min_size: int = 500, max_size: int = 1000) -> np.ndarray:
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

    def create_grid_pattern(self, shape: tuple) -> np.ndarray:
        """정확한 격자 패턴 생성"""
        height, width = shape
        grid = np.ones((height, width), dtype=np.uint8) * 255
        cell_height = height // 8
        cell_width = width // 8

        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 0:
                    y1 = i * cell_height
                    y2 = (i + 1) * cell_height
                    x1 = j * cell_width
                    x2 = (j + 1) * cell_width
                    grid[y1:y2, x1:x2] = 0

        return grid

    def normalize_checkerboard(self, image: np.ndarray) -> np.ndarray:
        """체커보드 이미지의 조명을 보정하고 이진화합니다."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 히스토그램 평활화 적용
        gray = cv2.equalizeHist(gray)

        # 노이즈 제거
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Otsu's 이진화
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 모폴로지 연산
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return binary

    def mouse_callback(self, event: int, x: int, y: int, flags: int, param: any) -> None:
        """마우스 클릭 이벤트 처리"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 최대 4개의 점만 저장
            if len(self.corner_points) < 4:
                self.corner_points.append((x, y))
                # 클릭한 위치에 원 그리기
                cv2.circle(self.image, (x, y), 5, (0, 0, 255), -1)
                # 점의 순서 표시
                cv2.putText(self.image, str(len(self.corner_points)), (x + 10, y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Select Corners', self.image)

                # 4개의 점이 모두 선택되면 변환 수행
                if len(self.corner_points) == 4:
                    transformed = self.transform_perspective()
                    cv2.imshow('Transformed Checkerboard', transformed)

    def get_corner_points(self, input_filename: str) -> None:
        """사용자로부터 코너 점 입력받기"""
        cv2.namedWindow('Select Corners')
        cv2.setMouseCallback('Select Corners', self.mouse_callback, input_filename)

        print("\n체커보드의 코너를 다음 순서로 클릭하세요:")
        print("1. 좌상단 -> 2. 우상단 -> 3. 우하단 -> 4. 좌하단")

        cv2.imshow('Select Corners', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def transform_perspective(self) -> np.ndarray:
        """선택된 점들을 기반으로 투시 변환 수행"""
        if len(self.corner_points) != 4:
            raise ValueError("4개의 코너 점이 필요합니다.")

        # 소스 포인트 (입력받은 코너 좌표)
        src_points = np.float32(self.corner_points)

        # 타겟 포인트 (정방형으로 변환될 좌표)
        dst_points = np.float32([
            [0, 0],  # 좌상단
            [self.output_size, 0],  # 우상단
            [self.output_size, self.output_size],  # 우하단
            [0, self.output_size]  # 좌하단
        ])

        # perspective transform 행렬 계산
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # 투시 변환 적용
        transformed = cv2.warpPerspective(
            self.original_image,
            matrix,
            (self.output_size, self.output_size)
        )

        # 이진화 적용
        binary = self.normalize_checkerboard(transformed)

        # 결과가 올바른지 확인하고 필요한 경우 반전
        white_pixels = cv2.countNonZero(binary)
        total_pixels = binary.shape[0] * binary.shape[1]

        # 흰색 픽셀이 전체의 60%를 넘으면 이미지를 반전
        if white_pixels > total_pixels * 0.6:
            binary = cv2.bitwise_not(binary)

        return binary


def main():
    # 명령행 인자 확인
    if len(sys.argv) != 2:
        print("사용법: python hw1_2.py <이미지파일>")
        print("예시: python hw1_2.py board2.jpg")
        sys.exit(1)

    # 이미지 파일 경로
    image_path = sys.argv[1]

    try:
        # 변환기 인스턴스 생성
        transformer = CheckerboardPerspectiveTransformer(image_path)

        # 코너 점 입력받기
        transformer.get_corner_points(image_path)

    except FileNotFoundError:
        print(f"에러: 파일을 찾을 수 없습니다 - {image_path}")
        sys.exit(1)
    except Exception as e:
        print(f"에러 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()