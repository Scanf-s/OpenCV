import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from PIL import Image
import sys
import os
import shutil
import matplotlib.pyplot as plt
import random
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

EPOCHS = 50
THRESHOLD = 0.75
BATCH_SIZE = 16

# 이미지 분류 예시 문서
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/
class EmpireStateModel:
    def __init__(self, model_path='empire_state_building_classification_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model()

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))

        self.model.to(self.device)
        self.model.eval()

        # https://pytorch.org/vision/stable/transforms.html#transforming-and-augmenting-images
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def _build_model():
        # https://pytorch.org/vision/stable/models.html#initializing-pre-trained-models
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        return model

    def calculate_metrics(self, loader):
        self.model.eval() # 평가 모드 설정
        all_preds = [] # 예측 결과 저장 리스트
        all_labels = [] # 실제 결과 저장 리스트

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)

                # 예측값을 threshold와 비교하여 이진 분류 수행
                # squeeze(): 차원 축소 함수 (불필요한 차원 제거)
                # threshold 값과 비교해서 True/False 반환한다.
                # float() 함수를 사용해서 1/0으로 변환
                predictions = (outputs.squeeze() >= THRESHOLD).float()

                # GPU 작업을 다시 CPU롷 변환해서 저장
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # https://driip.me/3ef36050-f5a3-41ea-9f23-874afe665342
        # 정확도 계산 -> 올바르게 예측한 샘플 수 / 전체 샘플 수
        accuracy = accuracy_score(all_labels, all_preds)

        # 정밀도 : Positive로 예측한 것 중에서 실제 positive인 비율
        # 재현율 : 실제 positive 중 positive인 것의 비율
        # F1 : precision과 recall의 조화 평균
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
        return accuracy, precision, recall, f1

    def train(self, train_dir, model_path='empire_state_building_classification_model.pth'):
        # 학습 데이터가 적기 때문에, 데이터 증강 설정을 해줘야한다.
        # https://teddylee777.github.io/pytorch/pytorch-image-transforms/
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.RandomAffine(degrees=20, translate=(0.15, 0.15), scale=(0.8, 1.2)),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 전체 데이터셋 로드
        full_dataset = ImageFolder(train_dir, transform=train_transform)

        # 전체 데이터를 8:2 비율로 훈련/검증 세트로 분할
        dataset_size = len(full_dataset)
        train_size = int(0.8 * dataset_size)

        # 인덱스를 무작위로 섞어서 분할
        indices = list(range(dataset_size))
        random.shuffle(indices)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        train_dataset = Subset(full_dataset, train_indices)
        train_labels = [full_dataset.targets[i] for i in train_indices]

        # 클래스별 인덱스 찾기
        positive_indices = [i for i, label in enumerate(train_labels) if label == full_dataset.class_to_idx['positive']]
        negative_indices = [i for i, label in enumerate(train_labels) if label == full_dataset.class_to_idx['negative']]

        # 더 적은 수의 클래스에 맞춰서 down sampling
        # 훈련 데이터의 클래스 불균형 해소 -> Positive 데이터 개수가 더 적기 때문에 맞춰줘야함
        min_samples = min(len(positive_indices), len(negative_indices))
        balanced_positive = random.sample(positive_indices, min_samples)
        balanced_negative = random.sample(negative_indices, min_samples)

        balanced_indices = balanced_positive + balanced_negative
        train_dataset = Subset(train_dataset, balanced_indices)

        # 검증 데이터는 원본 그대로 사용
        val_dataset = Subset(full_dataset, val_indices)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # 데이터가 이미 밸런싱되어 있으므로 기본 손실함수인 BCELoss 사용
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00005, weight_decay=1e-4)

        # Early stopping을 위한 변수, 모델 학습 곡선을 그리기 위한 변수 선언
        history = {'train_loss': [], 'val_loss': [], 'train_metrics': [], 'val_metrics': []}
        patience, trigger_times = 5, 0
        best_val_loss = float('inf')
        best_model_state = None

        for epoch in range(EPOCHS):
            self.model.train()
            running_train_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.float().to(self.device)
                optimizer.zero_grad() # 1. 그래디언트 초기화
                outputs = self.model(inputs) # 2. 순전파
                loss = criterion(outputs.squeeze(), labels) # 3. 손실 계산
                loss.backward() # 4. 역전파
                optimizer.step() # 5. 가중치 업데이트
                running_train_loss += loss.item()

            avg_train_loss = running_train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)

            # 검증 단계
            self.model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.float().to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs.squeeze(), labels)
                    running_val_loss += loss.item()

            avg_val_loss = running_val_loss / len(val_loader)
            history['val_loss'].append(avg_val_loss)

            # 성능 지표 계산
            train_metrics = self.calculate_metrics(train_loader)
            val_metrics = self.calculate_metrics(val_loader)
            history['train_metrics'].append(train_metrics)
            history['val_metrics'].append(val_metrics)

            print(f"Epoch {epoch + 1}")
            print(f"Train - Loss: {avg_train_loss:.4f}, Acc: {train_metrics[0]:.4f}, F1: {train_metrics[3]:.4f}")
            print(f"Val - Loss: {avg_val_loss:.4f}, Acc: {val_metrics[0]:.4f}, F1: {val_metrics[3]:.4f}")

            # Early stopping을 사용해서 overfitting 징조가 보이면 중단시킴
            # https://github.com/Bjarten/early-stopping-pytorch
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = self.model.state_dict().copy()
                torch.save(best_model_state, model_path)
                trigger_times = 0
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print("Early stopping. Best model state로 복구해서 모델 학습 중단")
                    self.model.load_state_dict(best_model_state)
                    break

        return history

    def detect(self, image_path, threshold=THRESHOLD):
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                prediction = self.model(image_tensor)
                print(f"예측값 : {prediction.item()}")
                return bool(prediction.item() >= threshold)

        except Exception as e:
            print(f"Error processing image: {e}")
            return False


def organize_dataset():
    # 데이터셋 ImageFolder에서 사용할 수 있도록 재구성
    os.makedirs('training_data/positive', exist_ok=True)
    os.makedirs('training_data/negative', exist_ok=True)
    os.makedirs('test_data/positive', exist_ok=True)
    os.makedirs('test_data/negative', exist_ok=True)

    for filename in os.listdir('empire'):
        src = os.path.join('empire', filename)
        dst = os.path.join('training_data/positive', filename)
        shutil.copy2(src, dst)

    for filename in os.listdir('empire_test'):
        src = os.path.join('empire_test', filename)
        dst = os.path.join('test_data/positive', filename)
        shutil.copy2(src, dst)

    for filename in os.listdir('not_empire'):
        src = os.path.join('not_empire', filename)
        dst = os.path.join('training_data/negative', filename)
        shutil.copy2(src, dst)

    for filename in os.listdir('not_empire_test'):
        src = os.path.join('not_empire_test', filename)
        dst = os.path.join('test_data/negative', filename)
        shutil.copy2(src, dst)


def print_dataset_info():
    # 데이터셋 개수 출력 함수
    train_pos = len(os.listdir('training_data/positive'))
    train_neg = len(os.listdir('training_data/negative'))
    test_pos = len(os.listdir('test_data/positive'))
    test_neg = len(os.listdir('test_data/negative'))

    print("Dataset:")
    print(f"학습용: {train_pos} positive, {train_neg} negative")
    print(f"테스트용: {test_pos} positive, {test_neg} negative")


def main():
    if len(sys.argv) != 2:
        print("실행 방법 : python hw2_2.py 이미지경로")
        sys.exit(1)

    test_image_path = sys.argv[1]
    model_path = 'empire_state_building_classification_model.pth'

    if not os.path.exists(model_path):
        organize_dataset()
        print_dataset_info()

        print("학습된 모델 파일이 존재하지 않으므로 학습을 수행합니다.")
        detector = EmpireStateModel(model_path)
        history = detector.train('training_data')

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label="Train Loss")
        plt.plot(history['val_loss'], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        train_f1 = [metrics[3] for metrics in history['train_metrics']]
        val_f1 = [metrics[3] for metrics in history['val_metrics']]
        plt.plot(train_f1, label="Train F1")
        plt.plot(val_f1, label="Validation F1")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.legend()

        plt.tight_layout()
        plt.savefig("training_metrics.png")
    else:
        detector = EmpireStateModel(model_path)
        result = detector.detect(test_image_path)
        print(result)


if __name__ == "__main__":
    main()