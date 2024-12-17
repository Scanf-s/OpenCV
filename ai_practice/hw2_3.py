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
        try:
            original_image = cv2.imread(image_path)
            if original_image is None:
                print("이미지를 불러올 수 없습니다.")
                return False

            pil_image = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

            target_layers = [self.model.layer4[-1]]
            cam = GradCAM(model=self.model, target_layers=target_layers)

            with torch.no_grad():
                prediction = self.model(image_tensor)
                is_empire = prediction.item() >= threshold

            if is_empire:
                grayscale_cam = cam(input_tensor=image_tensor)
                grayscale_cam = grayscale_cam[0]
                heatmap = cv2.resize(grayscale_cam, (original_image.shape[1], original_image.shape[0]))

                # 히트맵 임계값 조정
                binary_map = (heatmap > 0.4).astype(np.uint8)  # 임계값을 0.3으로 낮춤

                # 노이즈 제거를 위한 모폴로지 연산
                kernel = np.ones((5, 5), np.uint8)
                binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, kernel)
                binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_OPEN, kernel)

                contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    result_image = original_image.copy()
                    for contour in contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

                    cv2.imshow('Original Image', original_image)
                    cv2.imshow('Detection Result', result_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                    print(f"엠파이어 스테이트 빌딩이 감지되었습니다. (신뢰도: {prediction.item():.2f})")
                    return True

            cv2.imshow('Original Image', original_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            print(f"엠파이어 스테이트 빌딩이 감지되지 않았습니다. (신뢰도: {prediction.item():.2f})")
            return False

        except Exception as e:
            print(f"이미지 처리 중 오류 발생: {e}")
            return False


def main():
    if len(sys.argv) != 2:
        print("사용법: python hw2_3.py <image_path>")
        sys.exit(1)

    test_image_path = sys.argv[1]
    detector = EmpireStateDetector()
    detector.detect_and_visualize(test_image_path)


if __name__ == "__main__":
    main()