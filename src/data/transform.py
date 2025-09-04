# src/data/transform.py
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

def build_train_tf_base(img_size):
    transform = transforms.Compose([
        transforms.Resize(img_size),  # 이미지 크기를 img_size로 조정
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform

# 다른 transform 함수들도 필요하다면 img_size를 추가
def build_valid_tf(img_size):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform