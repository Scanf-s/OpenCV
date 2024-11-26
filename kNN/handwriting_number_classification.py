import sys  # 시스템 종료를 위해 사용
import numpy as np  # 배열 및 수학적 연산을 위한 라이브러리
import cv2  # OpenCV 라이브러리

# 마우스 이벤트를 처리하기 위한 변수 초기화
oldx, oldy = -1, -1


# 마우스 이벤트 콜백 함수 정의
def on_mouse(event, x, y, flags, _):
    global oldx, oldy  # 이전 좌표를 기록
    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스 왼쪽 버튼을 눌렀을 때
        oldx, oldy = x, y
    elif event == cv2.EVENT_LBUTTONUP:  # 마우스 왼쪽 버튼을 뗐을 때
        oldx, oldy = -1, -1
    elif event == cv2.EVENT_MOUSEMOVE:  # 마우스를 움직일 때
        if flags & cv2.EVENT_FLAG_LBUTTON:  # 왼쪽 버튼이 눌린 상태라면
            # 선을 그리며 필기
            cv2.line(img, (oldx, oldy), (x, y), (255, 255, 255), 40, cv2.LINE_AA)
            oldx, oldy = x, y  # 현재 좌표를 저장
            cv2.imshow('img', img)  # 이미지를 업데이트


# digits.png에서 학습 데이터와 레이블 생성
digits = cv2.imread('digits.png', cv2.IMREAD_GRAYSCALE)  # 흑백 이미지로 읽기

if digits is None:  # 이미지가 로드되지 않았을 경우
    print('Image load failed!')
    sys.exit()

h, w = digits.shape[:2]  # 이미지의 높이와 너비를 가져옴

# 이미지의 셀을 나누어 학습 데이터 생성
cells = [np.hsplit(row, w // 20) for row in np.vsplit(digits, h // 20)]
cells = np.array(cells)  # 셀을 numpy 배열로 변환
train_images = cells.reshape(-1, 400).astype(np.float32)  # 각 셀을 1차원 배열(400픽셀)로 변환
train_labels = np.repeat(np.arange(10), len(train_images) / 10)  # 각 셀에 해당하는 숫자 레이블 생성

# kNN 모델 생성 및 학습
knn = cv2.ml.KNearest_create()  # k-NN 모델 객체 생성
knn.train(train_images, cv2.ml.ROW_SAMPLE, train_labels)  # 학습 데이터로 모델 훈련

# 테스트 이미지 준비
img = np.zeros((400, 400), np.uint8)  # 400x400 크기의 검정색 이미지 생성
cv2.imshow('img', img)  # 초기 빈 이미지를 화면에 출력
cv2.setMouseCallback('img', on_mouse)  # 마우스 이벤트 처리 설정

# 테스트 루프
while True:
    c = cv2.waitKey()  # 키 입력 대기
    if c == 27:  # ESC 키를 누르면 종료
        break
    elif c == ord(' '):  # 스페이스바를 누르면 필기체를 예측
        # 이미지를 20x20 크기로 축소
        img_resize = cv2.resize(img, (20, 20), interpolation=cv2.INTER_AREA)
        img_flatten = img_resize.reshape(-1, 400).astype(np.float32)  # 1차원 배열로 변환

        # k-NN 모델로 예측
        _, res, _, _ = knn.findNearest(img_flatten, 3)  # k=3으로 예측
        print(int(res[0, 0]))  # 예측 결과 출력

        img.fill(0)  # 이미지 초기화
        cv2.imshow('img', img)  # 초기화된 이미지를 화면에 출력

cv2.destroyAllWindows()  # 모든 창 닫기
