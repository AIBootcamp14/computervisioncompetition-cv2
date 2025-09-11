import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
import timm
from typing import List, Dict, Tuple

# 1. train.csv에서 target이 3, 7인 이미지 ID 가져오기
def get_minor_class_ids(csv_path: str, classes: List[int]) -> pd.DataFrame:
    """주어진 클래스에 해당하는 데이터프레임을 필터링합니다."""
    df = pd.read_csv(csv_path)
    filtered_df = df[df['target'].isin(classes)].copy()
    filtered_df['target'] = filtered_df['target'].apply(lambda x: 0 if x == 3 else 1)
    print(f"클래스 {classes}에 해당하는 데이터 {len(filtered_df)}개를 찾았습니다.")
    return filtered_df

# 2. 이미지 증강
def get_transforms() -> transforms.Compose:
    """vit 모델에 맞는 이미지 변환 파이프라인을 정의합니다."""
    return transforms.Compose([
        transforms.Resize((512, 512)), # vit 모델에 맞는 이미지 크기 조정
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomRotation(degrees=(0, 180)),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

# 3. 데이터셋 클래스
class CustomMinorClassDataset(Dataset):
    """3, 7번 클래스 데이터셋을 위한 커스텀 클래스입니다."""
    def __init__(self, df: pd.DataFrame, img_dir: str, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.image_ids = self.df['ImageID'].tolist() if 'ImageID' in self.df.columns else self.df['ID'].tolist()
        self.labels = self.df['target'].tolist() if 'target' in self.df.columns else [-1] * len(self.df)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_id = self.image_ids[idx]
        file_name = f"{image_id}.jpg" if not image_id.endswith(".jpg") else image_id
        img_path = os.path.join(self.img_dir, file_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"파일을 찾을 수 없습니다: {img_path}. 다음 파일로 넘어갑니다.")
            return self.__getitem__((idx + 1) % len(self.df))
        
        if self.transform:
            image = self.transform(image)
            
        label = self.labels[idx]
        return image, label

# 4. 모델 학습
def train_minor_classifier(df: pd.DataFrame, img_dir: str, device: torch.device) -> nn.Module:
    """보조 모델을 학습하고 반환합니다."""
    dataset = CustomMinorClassDataset(df, img_dir, transform=get_transforms())
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    
    # 미리 학습된 vit_base 모델을 로드
    model = timm.create_model('vit_base_patch16_siglip_512.webli', pretrained=True, num_classes=2).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    print("3, 7번 클래스 전용 모델 학습 시작 (Vit-base 사용)...")
    num_epochs = 20
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")
    print("학습 완료.")
    return model

# 5. 테스트셋에 대한 추론
def predict_test_set_for_classes(model: nn.Module, test_df: pd.DataFrame, img_dir: str, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """테스트 데이터에 대한 추론을 수행합니다."""
    transform = get_transforms()
    test_dataset = CustomMinorClassDataset(test_df, img_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    model.eval()
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for inputs, _ in tqdm(test_loader, desc="추론 진행 중"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_preds), np.array(all_probs)

# 6. 기존 예측에 확률 기반으로 덮어쓰기
def integrate_new_predictions(base_submission_df: pd.DataFrame, minor_test_df: pd.DataFrame, new_preds: np.ndarray, new_probs: np.ndarray) -> pd.DataFrame:
    """
    새로운 예측 결과에 신뢰도를 적용하여 기존 제출 파일에 통합합니다.
    """
    base_submission_df.set_index('ID', inplace=True)
    minor_test_df.set_index('ID', inplace=True)
    minor_class_ids = minor_test_df.index.tolist()
    
    PROBABILITY_THRESHOLD = 0.95
    
    for i, img_id in enumerate(minor_class_ids):
        prob_of_predicted_class = new_probs[i, new_preds[i]]
        
        if prob_of_predicted_class > PROBABILITY_THRESHOLD:
            predicted_class_val = 3 if new_preds[i] == 0 else 7
            base_submission_df.loc[img_id, 'target'] = predicted_class_val
    
    base_submission_df.reset_index(inplace=True)
    print("새로운 예측 결과로 기존 제출 파일 업데이트 완료.")
    return base_submission_df

# 메인 실행 함수
def run_minor_class_pipeline():
    """3, 7번 클래스 보정 파이프라인을 실행합니다."""
    ROOT_DIR = "."
    TRAIN_CSV_PATH = os.path.join(ROOT_DIR, 'data', 'train.csv')
    TRAIN_IMG_DIR = os.path.join(ROOT_DIR, 'data', 'train')
    TEST_CSV_PATH = os.path.join(ROOT_DIR, 'output', 'vit_base_patch16_siglip_512.webli_auto.csv')
    TEST_IMG_DIR = os.path.join(ROOT_DIR, 'data', 'test')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 훈련 데이터 준비
    minor_train_df = get_minor_class_ids(TRAIN_CSV_PATH, [3, 7])
    
    # 2. 전용 모델 학습
    minor_classifier_model = train_minor_classifier(minor_train_df, TRAIN_IMG_DIR, device)
    
    # 3. 테스트 데이터 준비 (기존 모델이 3, 7로 예측한 데이터만 추출)
    test_df_from_base_model = pd.read_csv(TEST_CSV_PATH)
    minor_test_df = test_df_from_base_model[test_df_from_base_model['target'].isin([3, 7])].copy()
    
    # 4. 테스트셋에 대한 추론
    new_preds, new_probs = predict_test_set_for_classes(minor_classifier_model, minor_test_df, TEST_IMG_DIR, device)
    
    # 5. 기존 제출 파일에 새로운 예측 결과 통합
    final_submission_df = pd.read_csv(TEST_CSV_PATH)
    final_submission_df = integrate_new_predictions(final_submission_df, minor_test_df, new_preds, new_probs)
    
    # 최종 파일 저장
    final_submission_path = os.path.join(os.path.dirname(TEST_CSV_PATH), "final_submission_with_minor_correction.csv")
    final_submission_df.to_csv(final_submission_path, index=False)
    print(f"최종 제출 파일이 {final_submission_path}에 저장되었습니다.")

if __name__ == "__main__":
    run_minor_class_pipeline()