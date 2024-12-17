import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
import sys

from hw2_2 import EmpireStateModel

THRESHOLD = 0.75


class EmpireStateDetector(EmpireStateModel):
    def __init__(self, model_path='empire_state_building_classification_model.pth'):
        """
        hw2_2.py에서 만든 모델을 사용하여 Empire state building인지 예측해야 한다.
        """
        super().__init__(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model()

        if not torch.cuda.is_available():
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
        else:
            self.model.load_state_dict(torch.load(model_path, weights_only=True))

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def detect_and_visualize(self, image_path, threshold=THRESHOLD):
        """
        입력된 이미지를 분석해서 Empire State Building이 감지되면 Grad-CAM을 사용해서 탐지 영역을 시각화하고,
        해당 영역에 빨간색 경계선 박스를 그려주는 함수.
        # https://sotudy.tistory.com/42
        # https://github.com/jacobgil/pytorch-grad-cam
        # https://ai-bt.tistory.com/entry/Class-Activation-Mapping-CAM%EA%B3%BC-Grad-CAM
        """
        try:
            original_image = cv2.imread(image_path)
            if original_image is None:
                print("이미지를 불러올 수 없습니다.")
                return False

            pil_image = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

            target_layers = [self.model.layer4[-1]] # 학습 시킨 모델의 최종 레이어를 수정해서 Empire state building 분류를 학습시켰으므로 해당 레이어를 타겟으로 설정
            cam = GradCAM(model=self.model, target_layers=target_layers) # 타겟 레이어의 활성화 맵을 기반으로 중요 영역을 시각화하는 클래스

            with torch.no_grad():
                # 이미지에 대해 Empire state building인지 예측 수행
                prediction = self.model(image_tensor)
                is_empire = prediction.item() >= threshold

            if is_empire:
                grayscale_cam = cam(input_tensor=image_tensor) # 만약 탐지되었다면 cam을 사용해서 히트맵 생성
                grayscale_cam = grayscale_cam[0]
                heatmap = cv2.resize(grayscale_cam, (original_image.shape[1], original_image.shape[0]))

                # 히트맵 임계값 조정
                binary_map = (heatmap > 0.4).astype(np.uint8)

                # 노이즈 제거를 위한 모폴로지 연산
                kernel = np.ones((5, 5), np.uint8)
                binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, kernel)
                binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_OPEN, kernel)

                # 경계선 탐색 및 빨간 박스 그리기
                contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    result_image = original_image.copy()
                    for contour in contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

                    cv2.imshow('Detected', result_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    return True

            cv2.imshow('Not Detected', original_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return False

        except Exception as e:
            print(f"이미지 처리 중 오류 발생: {e}")
            return False


def main():
    if len(sys.argv) != 2:
        print("사용법: python hw2_3.py 이미지경로")
        sys.exit(1)

    test_image_path = sys.argv[1]
    detector = EmpireStateDetector()
    detector.detect_and_visualize(test_image_path)


if __name__ == "__main__":
    main()