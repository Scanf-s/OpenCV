import numpy as np
import cv2

# 학습 데이터 및 레이블 리스트와 K 값 초기화
train = []  # 학습 데이터 포인트를 저장
label = []  # 각 데이터 포인트의 레이블을 저장
k_value = 1  # 초기 K 값


# K 값이 변경될 때 호출되는 콜백 함수
def on_k_changed(pos):
    global k_value  # 전역 변수 k_value를 수정
    k_value = pos  # 트랙바에서 현재 값을 가져옴
    if k_value < 1:  # k 값은 최소 1 이상이어야 함
        k_value = 1
    trainAndDisplay()  # 새로운 k 값으로 결과 업데이트


# 새로운 학습 데이터를 추가하는 함수
def addPoint(x, y, c):
    train.append([x, y])  # 학습 데이터에 새로운 포인트 추가
    label.append([c])  # 해당 포인트의 레이블 추가


# KNN 학습 및 결과 시각화 함수
def trainAndDisplay():
    global img  # 이미지를 수정하므로 전역 변수 사용
    train_array = np.array(train).astype(np.float32)  # 학습 데이터를 float32로 변환
    label_array = np.array(label)  # 레이블 데이터를 numpy 배열로 변환
    knn.train(train_array, cv2.ml.ROW_SAMPLE, label_array)  # KNN 학습 수행

    # 이미지의 각 픽셀을 순회하며 분류
    for j in range(img.shape[0]):  # 높이
        for i in range(img.shape[1]):  # 너비
            sample = np.array([[i, j]]).astype(np.float32)  # 현재 픽셀 좌표를 샘플로 변환
            ret, res, _, _ = knn.findNearest(sample, k_value)  # KNN으로 분류
            response = int(res[0, 0])  # 예측 결과를 정수형으로 변환

            # 예측 결과에 따라 픽셀 색상을 설정
            if response == 0:
                img[j, i] = (128, 128, 255)  # 레이블 0 -> 파란색 계열
            elif response == 1:
                img[j, i] = (128, 255, 128)  # 레이블 1 -> 초록색 계열
            elif response == 2:
                img[j, i] = (255, 128, 128)  # 레이블 2 -> 빨간색 계열

    # 학습 데이터 포인트를 이미지에 점으로 표시
    for i in range(len(train)):
        x, y = train[i]
        l = label[i][0]
        if l == 0:
            cv2.circle(img, (x, y), 5, (0, 0, 128), -1, cv2.LINE_AA)  # 레이블 0 -> 파란 점
        elif l == 1:
            cv2.circle(img, (x, y), 5, (0, 128, 0), -1, cv2.LINE_AA)  # 레이블 1 -> 초록 점
        elif l == 2:
            cv2.circle(img, (x, y), 5, (128, 0, 0), -1, cv2.LINE_AA)  # 레이블 2 -> 빨간 점

    cv2.imshow('knn', img)  # 결과 이미지를 화면에 출력


# 빈 이미지와 KNN 객체 초기화
img = np.zeros((500, 500, 3), np.uint8)  # 검정색 배경 이미지
knn = cv2.ml.KNearest_create()  # KNN 객체 생성

# 트랙바와 윈도우 설정
cv2.namedWindow('knn')  # 윈도우 창 생성
cv2.createTrackbar('k_value', 'knn', k_value, 5, on_k_changed)  # 트랙바 생성

# 난수를 사용해 학습 데이터 생성
NUM = 30  # 각 클래스의 데이터 포인트 수
rn = np.zeros((NUM, 2), np.int32)  # 난수를 저장할 배열

# 클래스 0의 데이터 생성
cv2.randn(rn, 0, 50)  # 평균 0, 표준편차 50의 정규분포 난수 생성
for i in range(NUM):
    addPoint(rn[i, 0] + 150, rn[i, 1] + 150, 0)  # (150, 150) 기준

# 클래스 1의 데이터 생성
cv2.randn(rn, 0, 50)
for i in range(NUM):
    addPoint(rn[i, 0] + 350, rn[i, 1] + 150, 1)  # (350, 150) 기준

# 클래스 2의 데이터 생성
cv2.randn(rn, 0, 70)
for i in range(NUM):
    addPoint(rn[i, 0] + 250, rn[i, 1] + 400, 2)  # (250, 400) 기준

# 학습 및 결과 표시
trainAndDisplay()

# 결과를 대기하며 표시
cv2.imshow('knn', img)  # 최종 결과 출력
cv2.waitKey()  # 키 입력 대기
cv2.destroyAllWindows()  # 모든 창 닫기
