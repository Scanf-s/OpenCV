import numpy as np
import cv2
import struct
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import time
from sklearn.preprocessing import StandardScaler


class FeatureExtractor:
    def __init__(self, method='HOG'):
        self.method = method

    def extract_hog(self, img):
        # HOG 특징 추출 방법
        win_size = (28, 28)
        cell_size = (4, 4)
        block_size = (8, 8)
        block_stride = (4, 4)
        num_bins = 9

        hog = cv2.HOGDescriptor(win_size, block_size, block_stride,
                                cell_size, num_bins)
        return hog.compute(img).flatten()

    def extract_sift(self, img):
        # SIFT 특징 추출 방법
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(img, None)
        if descriptors is None:
            return np.zeros(128)  # SIFT 특징의 기본 크기
        return descriptors.flatten()[:128]  # 첫 128개 특징만 사용

    # https://076923.github.io/posts/Python-opencv-38/
    def extract_orb(self, img):
        # ORB 파라미터 설정
        orb = cv2.ORB_create(
            nfeatures=32,  # 특징점 개수
            scaleFactor=1.2,  # 스케일 factor
            nlevels=8,  # 피라미드 레벨 수
            edgeThreshold=2,  # 엣지 임계값
            firstLevel=0,  # 첫 번째 레벨
            WTA_K=2,  # 방향성을 계산하는데 사용되는 점의 수
            patchSize=31,  # 패치 크기
            fastThreshold=20  # FAST 임계값
        )

        # 이미지 전처리
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # 특징점과 디스크립터 추출
        keypoints = orb.detect(img, None)
        # 특징점을 고르게 분포시키기 위해 이미지를 그리드로 나누어 선택
        grid_size = 7  # 7x7 그리드
        height, width = img.shape
        cell_height = height // grid_size
        cell_width = width // grid_size

        selected_keypoints = []
        for i in range(grid_size):
            for j in range(grid_size):
                cell_keypoints = [kp for kp in keypoints if
                                  i * cell_height <= kp.pt[1] < (i + 1) * cell_height and
                                  j * cell_width <= kp.pt[0] < (j + 1) * cell_width]
                if cell_keypoints:
                    # 각 셀에서 가장 강한 특징점 선택
                    cell_keypoints.sort(key=lambda x: x.response, reverse=True)
                    selected_keypoints.append(cell_keypoints[0])

        # 특징점이 충분하지 않은 경우 처리
        if len(selected_keypoints) < 32:
            # 부족한 특징점을 이미지 중심에서 방사형으로 추가
            center_x, center_y = width // 2, height // 2
            for r in range(10, max(width, height), 10):
                for theta in range(0, 360, 45):
                    if len(selected_keypoints) >= 32:
                        break
                    x = center_x + int(r * np.cos(np.radians(theta)))
                    y = center_y + int(r * np.sin(np.radians(theta)))
                    if 0 <= x < width and 0 <= y < height:
                        kp = cv2.KeyPoint(x, y, 31)
                        selected_keypoints.append(kp)

        # 디스크립터 계산
        keypoints, descriptors = orb.compute(img, selected_keypoints)

        if descriptors is None:
            return np.zeros(256, dtype=np.uint8)

        # 디스크립터를 일정한 크기로 만들기
        feature_vector = np.zeros(256, dtype=np.uint8)
        descriptors_flattened = descriptors.flatten()
        feature_vector[:min(256, len(descriptors_flattened))] = descriptors_flattened[
                                                                :min(256, len(descriptors_flattened))]

        return feature_vector

    def extract_features(self, img):
        if self.method == 'HOG':
            return self.extract_hog(img)
        elif self.method == 'SIFT':
            return self.extract_sift(img)
        elif self.method == 'ORB':
            return self.extract_orb(img)
        else:
            raise ValueError(f"Unsupported feature extraction method: {self.method}")


class DigitClassifier:
    def __init__(self, model_type='kNN', feature_type='HOG'):
        self.feature_extractor = FeatureExtractor(feature_type)
        self.model = self._get_model(model_type)
        self.scaler = StandardScaler()

    def _get_model(self, model_type):
        if model_type == 'kNN':
            if self.feature_extractor.method == 'ORB':
                # ORB의 경우 Hamming distance 사용
                return KNeighborsClassifier(n_neighbors=3, metric='hamming')
            else:
                return KNeighborsClassifier(n_neighbors=3)
        elif model_type == 'SVM':
            if self.feature_extractor.method == 'ORB':
                # ORB의 경우 다른 커널 사용
                return SVC(kernel='linear', C=1.0)
            else:
                return SVC(kernel='rbf', C=1.0)
        else:
            raise ValueError(f"모델 타입이 올바르지 않습니다 : {model_type}")

    def preprocess(self, img):
        # 이미지 전처리
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (28, 28))
        return img

    def extract_features_batch(self, images):
        features = []
        for img in images:
            img = self.preprocess(img)
            features.append(self.feature_extractor.extract_features(img))
        return np.array(features)

    def fit(self, X, y):
        print("특징 추출 시작...")
        start_time = time.time()
        X_features = self.extract_features_batch(X)
        feature_time = time.time() - start_time

        print("특징 스케일링...")
        X_features = self.scaler.fit_transform(X_features)

        print("모델 학습 시작...")
        start_time = time.time()
        self.model.fit(X_features, y)
        train_time = time.time() - start_time

        print(f"특징 추출 시간: {feature_time:.2f}초")
        print(f"학습 시간: {train_time:.2f}초")

    def predict(self, X):
        X_features = self.extract_features_batch(X)
        X_features = self.scaler.transform(X_features)
        start_time = time.time()
        predictions = self.model.predict(X_features)
        inference_time = time.time() - start_time
        print(f"추론 시간: {inference_time:.2f}초")
        return predictions


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    print(f"정확도: {accuracy:.4f}")
    print("\n분류 리포트:")
    print(report)


def load_images(file_path):
    with open(file_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
    return images


def load_labels(file_path):
    with open(file_path, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels


def load_mnist():
    # MNIST 데이터 로드
    train_images_path = "train-images.idx3-ubyte"
    train_labels_path = "train-labels.idx1-ubyte"
    test_images_path = "t10k-images.idx3-ubyte"
    test_labels_path = "t10k-labels.idx1-ubyte"

    X_train = load_images(train_images_path)
    y_train = load_labels(train_labels_path)
    X_test = load_images(test_images_path)
    y_test = load_labels(test_labels_path)

    return X_train, X_test, y_train, y_test


# 사용 예시
def main():
    # MNIST 데이터 로드
    X_train, X_test, y_train, y_test = load_mnist()
    models = ['kNN', 'SVM']
    features = ['HOG', 'SIFT', 'ORB']  # ORB 추가

    results = {}
    # 모든 모델과 특징 추출 방법 조합을 테스트
    for model_type in models:
        for feature_type in features:
            print(f"\n{model_type} with {feature_type} 실험")
            classifier = DigitClassifier(model_type, feature_type)

            # 학습
            classifier.fit(X_train, y_train)

            # 평가
            evaluate_model(classifier, X_test, y_test)

            # 결과 저장
            results[f"{model_type}_{feature_type}"] = {
                "accuracy": accuracy_score(y_test, classifier.predict(X_test))
            }

    return results


if __name__ == "__main__":
    main()