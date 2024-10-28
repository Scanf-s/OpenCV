import cv2
import numpy as np
import sys

def count_checker_pieces(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: 이미지 파일 '{image_path}'을(를) 읽을 수 없습니다.")
        sys.exit(1)

    # 이미지를 그레이스케일로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 가우시안 블러 적용
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # 에지 검출
    edges = cv2.Canny(blurred, 10, 90)

    # 허프 변환 원 검출
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=0.95,
        minDist=15,
        param1=100,
        param2=20,
        minRadius=5,
        maxRadius=21
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

        # 검출된 영역의 색상 평균으로 계산하는것 보다,
        # 그냥 중심점의 색상 찍어서 가져오는게 더 정확한 결과가 나오길래
        # 해당 방식을 선택했음

        # 중심점의 색상 가져오기
        center_color = image[center[1], center[0]]

        # BGR을 그레이스케일 밝기로 변환
        center_intensity = 0.299 * center_color[2] + 0.587 * center_color[1] + 0.114 * center_color[0]

        # 밝기 값을 기준으로 말의 색상 분류
        intensity_threshold = 130  # 임계값 조정 (기존 160 -> 130)
        if center_intensity >= intensity_threshold:
            color = (0, 255, 0)  # 녹색 원 (밝은 색 말)
            light_count += 1
        else:
            color = (0, 0, 255)  # 빨간색 원 (어두운 색 말)
            dark_count += 1

        # 원의 경계 그리기
        cv2.circle(output, center, radius, color, 2)
        # 중심점 그리기
        cv2.circle(output, center, radius, color, 2)

    # 결과 출력
    print(f"밝은 색 말의 수: {light_count}")
    print(f"어두운 색 말의 수: {dark_count}")

    # 결과 이미지 표시
    cv2.imshow('Detected Pieces', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    if len(sys.argv) != 2:
        print("사용방법: python script.py <image_path>")
        print("예시: python script.py checkerboard.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    count_checker_pieces(image_path)

if __name__ == "__main__":
    main()
