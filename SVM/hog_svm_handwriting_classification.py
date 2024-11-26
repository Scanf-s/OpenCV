import cv2  # OpenCV 라이브러리
import numpy as np  # 배열 연산을 위한 numpy

# 1. digits.png 이미지를 읽고 전처리
digits = cv2.imread('digits.png', cv2.IMREAD_GRAYSCALE)  # 흑백 이미지로 읽기

if digits is None:
    print("Image not found!")
    exit()

# digits.png는 5000개의 숫자로 구성된 2000x1000 크기의 이미지
# 각 숫자의 크기는 20x20이므로 이를 분할
h, w = digits.shape[:2]  # 이미지의 높이와 너비 가져오기
cells = [np.hsplit(row, w // 20) for row in np.vsplit(digits, h // 20)]  # 20x20 크기로 분할
cells = np.array(cells)  # 리스트를 numpy 배열로 변환

# 2. 데이터를 학습용과 테스트용으로 나눔
# 학습 데이터는 앞 50개의 숫자(0~9에 대해 각 숫자당 50개)
train_cells = cells[:, :50].reshape(-1, 400).astype(np.float32)  # 학습 데이터
test_cells = cells[:, 50:].reshape(-1, 400).astype(np.float32)  # 테스트 데이터

# 3. HOGDescriptor 객체 생성
# HOG 특징 벡터를 추출하기 위한 설정
hog = cv2.HOGDescriptor((20, 20), (10, 10), (5, 5), (5, 5), 9)

# 4. HOG 특징 벡터 계산 함수
def compute_hog_features(cells):
    hog_features = []
    for cell in cells:
        hog_feature = hog.compute(cell.reshape(20, 20))  # HOG 계산
        hog_features.append(hog_feature)
    return np.array(hog_features, dtype=np.float32).reshape(-1, 324)  # 324차원의 HOG 벡터

# 5. 학습 데이터와 테스트 데이터에서 HOG 특징 벡터 추출
train_hog = compute_hog_features(train_cells.astype(np.uint8))  # 학습 데이터의 HOG 벡터
test_hog = compute_hog_features(test_cells.astype(np.uint8))  # 테스트 데이터의 HOG 벡터

# 6. 라벨 생성
# 학습 데이터와 테스트 데이터의 라벨 생성 (0~9 각 숫자에 대해 50개씩)
train_labels = np.repeat(np.arange(10), train_hog.shape[0] // 10).astype(np.int32)  # HOG 샘플 수에 맞춰 라벨 생성
test_labels = np.repeat(np.arange(10), 50).astype(np.int32)  # 테스트 데이터 라벨
train_labels = train_labels.reshape(-1, 1)

print(f"train_hog.shape: {train_hog.shape}")  # HOG 특징 벡터의 크기
print(f"train_labels.shape: {train_labels.shape}")  # 라벨 벡터의 크기

# 7. SVM 모델 생성 및 설정
svm = cv2.ml.SVM_create()  # SVM 모델 생성
svm.setKernel(cv2.ml.SVM_RBF)  # RBF 커널 사용
svm.setType(cv2.ml.SVM_C_SVC)  # C-SVC 유형 설정
svm.setC(2.5)  # 정규화 상수 C 설정
svm.setGamma(0.02)  # RBF 커널의 gamma 설정

# 8. SVM 모델 학습
svm.train(train_hog, cv2.ml.ROW_SAMPLE, train_labels)  # 학습 수행

# 9. SVM 모델 저장
svm.save('hog_svm_digits_model.yml')  # 학습된 모델 저장

# 10. SVM 모델 테스트
_, predictions = svm.predict(test_hog)  # 테스트 데이터 예측
accuracy = np.mean(predictions == test_labels) * 100  # 정확도 계산

print(f"SVM Accuracy: {accuracy:.2f}%")  # 정확도 출력

# 11. 사용자 입력 필기체 숫자 인식
# 빈 이미지를 생성하고 마우스 이벤트를 통해 숫자를 필기
def on_mouse(event, x, y, flags, param):
    global old_x, old_y
    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스 왼쪽 버튼 클릭
        old_x, old_y = x, y
    elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:  # 드래그 중일 때
        cv2.line(drawing, (old_x, old_y), (x, y), 255, 5)  # 숫자를 흰색으로 그림
        old_x, old_y = x, y
        cv2.imshow("Input", drawing)
    elif event == cv2.EVENT_LBUTTONUP:  # 마우스 버튼 떼기
        old_x, old_y = -1, -1

# 빈 캔버스 생성
drawing = np.zeros((400, 400), dtype=np.uint8)
old_x, old_y = -1, -1
cv2.imshow("Input", drawing)
cv2.setMouseCallback("Input", on_mouse)

while True:
    key = cv2.waitKey(1)
    if key == 27:  # ESC 키 입력 시 종료
        break
    elif key == ord(' '):  # 스페이스바 입력 시 숫자 인식
        resized = cv2.resize(drawing, (20, 20), interpolation=cv2.INTER_AREA)  # 20x20 크기로 리사이즈
        hog_feature = hog.compute(resized).reshape(1, -1)  # HOG 특징 벡터 계산
        _, result = svm.predict(hog_feature)  # SVM으로 예측
        print(f"Predicted Number: {int(result[0, 0])}")  # 예측된 숫자 출력
        drawing.fill(0)  # 캔버스 초기화
        cv2.imshow("Input", drawing)

cv2.destroyAllWindows()  # 창 닫기
