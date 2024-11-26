import numpy as np  # 배열 및 수학적 계산을 위한 라이브러리
import cv2  # OpenCV 라이브러리

# 학습 데이터와 레이블 정의
train = np.array([[150, 200], [200, 250], [100, 250], [150, 300],
                  [350, 100], [400, 200], [350, 300], [350, 400]], dtype=np.float32)
label = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)

# SVM 모델 생성 및 설정
svm = cv2.ml.SVM_create()  # SVM 객체 생성
svm.setType(cv2.ml.SVM_C_SVC)  # C-Support Vector Classification 설정
svm.setKernel(cv2.ml.SVM_RBF)  # 커널을 RBF(Radial Basis Function)로 설정
# svm.setKernel(cv2.ml.SVM_LINEAR)  # 주석 처리: 선형 커널을 사용하려면 활성화
svm.trainAuto(train, cv2.ml.ROW_SAMPLE, label)  # 자동 하이퍼파라미터 튜닝 및 학습 수행

# 분류 결과를 시각화할 빈 이미지 생성
img = np.zeros((500, 500, 3), dtype=np.uint8)  # 검정색 배경의 500x500 이미지 생성

# 각 픽셀에 대해 SVM 예측 수행
for j in range(img.shape[0]):  # 이미지의 높이 순회
    for i in range(img.shape[1]):  # 이미지의 너비 순회
        test = np.array([[i, j]], dtype=np.float32)  # 현재 픽셀의 좌표를 테스트 데이터로 변환
        _, res = svm.predict(test)  # SVM 모델로 분류 수행
        if res == 0:
            img[j, i] = (128, 128, 255)  # 클래스 0: 파란색
        elif res == 1:
            img[j, i] = (128, 255, 128)  # 클래스 1: 초록색

# 학습 데이터 포인트를 이미지에 시각적으로 표시
color = [(0, 0, 128), (0, 128, 0)]  # 클래스별 포인트 색상 설정
for i in range(train.shape[0]):
    x = int(train[i, 0])  # 학습 데이터 x 좌표
    y = int(train[i, 1])  # 학습 데이터 y 좌표
    l = label[i]  # 해당 포인트의 클래스 레이블
    cv2.circle(img, (x, y), 5, color[l], -1, cv2.LINE_AA)  # 학습 데이터 포인트를 그림

# 결과를 화면에 출력
cv2.imshow('svm', img)  # SVM 분류 결과 시각화
cv2.waitKey()  # 키 입력 대기
cv2.destroyAllWindows()  # 모든 창 닫기