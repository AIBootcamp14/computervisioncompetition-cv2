import cv2
import random
import torch
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2


# set resolution size
IMG_SIZE = 512  # 512, 576, 608, 640


def _finalize_to_square(size = IMG_SIZE):
    # fit img size square wrt longer side, pad the shorter side
    return [
        A.LongestMaxSize(max_size=size, always_apply=True),
        A.PadIfNeeded(min_height=size, min_width=size, border_mode=0, value=255, always_apply=True)
    ]


# train data : augmentation -> square화 -> normalize & tensorize
def build_train_tf_base(size, mean, std):
    return A.Compose([
        A.OneOf([A.RandomRotate90(), A.Rotate(limit=10, border_mode=0, value=255, p=0.5)]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.GaussNoise(var_limit=(5.0, 15.0), p=0.4),
        A.ImageCompression(quality_lower=30, quality_upper=90, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.3),

        # square화
        *_finalize_to_square(size),

        # normalize & tensorize
        # A.Normalize(mean=(0.5,)*3, std=(0.5,)*3),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])


# train data (minor class): heavy augmentation & include affine -> square화 -> normalize & tensorize
def build_train_tf_heavy(size, mean, std):
    return A.Compose([
        A.OneOf([A.RandomRotate90(p=1.0), A.Rotate(limit=25, border_mode=0, value=255)], p=0.8),
        A.HorizontalFlip(p=0.6),
        A.VerticalFlip(p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(5.0, 30.0)),
            A.ISONoise(intensity=(0.1, 0.5)),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1)),
        ], p=0.6),
        A.OneOf([A.MotionBlur(blur_limit=3), A.MedianBlur(blur_limit=3), A.GaussianBlur(blur_limit=3)], p=0.3),
        A.ImageCompression(quality_lower=30, quality_upper=80, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.2, p=0.4),

        # Affine for minor class (minor class are easily overfitted)
        A.Affine(scale=(0.95,1.05), shear=(-4,4), translate_percent=(0.0,0.02),
                 fit_output=True, mode=0, cval=255, p=0.3),

        # square화
        *_finalize_to_square(size),

        # normalize & tensorize
        # A.Normalize(mean=(0.5,)*3, std=(0.5,)*3),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])


############ 문서류에 추가해볼 것 -> 다른 샘플을 직각 회전/축소or확대해서 반투명 오버레이 ############
class OverlayAnotherDoc(A.ImageOnlyTransform):
    def __init__(self, alpha=(0.25, 0.5), p=0.2):
        super().__init__(p=p)
        self.alpha = alpha
    def apply(self, img, **params):
        h, w = img.shape[:2]
        base = img.copy()
        angle = np.random.choice([0, 90, 180, 270])
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, np.random.uniform(0.9, 1.1))
        over = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
        a = np.random.uniform(*self.alpha)
        return cv2.addWeighted(base, 1.0, over, a, 0.0)
################################################################################################


# # Mixup 추가
# def mixup_collate(batch, alpha = 0.2, apply_prob= 0.5):
#     images = torch.stack([b[0] for b in batch], dim=0)
#     labels = torch.tensor([b[1] for b in batch], dtype=torch.long)

#     B = images.size(0)
#     if B < 2 or random.random() > apply_prob or alpha <= 0:
#         return images, labels, None, 1.0

#     lam = torch.distributions.Beta(alpha, alpha).sample().item()
#     perm = torch.randperm(B)

#     x1 = images
#     x2 = images[perm]

#     images = lam * x1 + (1 - lam) * x2
#     return images, labels, labels[perm], lam


# 이미지 클래스별로 증강 기법을 다르게 적용하기
# 문서류
def make_tf_doc(size, mean, std):
    return A.Compose([
        # 회전/기울이기
        A.Affine(rotate=(-12, 12), shear=(-5, 5),
                 translate_percent=(0.0, 0.03),
                 fit_output=True, mode=0, cval=255, p=0.25),

        A.ToGray(p=0.6),
        A.InvertImg(p=0.15),
        # 채도 변화
        A.HueSaturationValue(hue_shift_limit=4, sat_shift_limit=10, val_shift_limit=10, p=0.25),
        # 대비 변화
        A.CLAHE(clip_limit=(1, 3), p=0.25),
        # 저해상/압축
        A.Downscale(scale_min=0.7, scale_max=0.9, interpolation=cv2.INTER_LINEAR, p=0.2),
        A.ImageCompression(quality_lower=20, quality_upper=85, p=0.25),
        A.GaussNoise(var_limit=(5.0, 15.0), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.3),
        A.ToGray(p=0.3),

        OverlayAnotherDoc(p=0.2),

        *_finalize_to_square(size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])


# 신분증/증서류
def make_tf_idcert(size, mean, std):
    return A.Compose([
        A.Perspective(scale=(0.02, 0.06), pad_mode=0, pad_val=255, p=0.25),
        A.Affine(scale=(0.97, 1.03), rotate=(-6, 6), shear=(-4, 4),
                 translate_percent=(0.0, 0.02),
                 fit_output=True, mode=0, cval=255, p=0.35),
        # 가리기
        A.CoarseDropout(max_holes=3, max_height=int(size*0.12), max_width=int(size*0.12),
                        fill_value=255, p=0.2),
        # 채도 변화
        A.HueSaturationValue(hue_shift_limit=4, sat_shift_limit=12, val_shift_limit=12, p=0.25),
        # 샤픈으로 미세 텍스트 엣지 강화
        A.Sharpen(alpha=(0.05, 0.15), lightness=(0.9, 1.1), p=0.2),
        # 저해상/압축
        A.Downscale(scale_min=0.7, scale_max=0.9, interpolation=cv2.INTER_LINEAR, p=0.2),
        A.ImageCompression(quality_lower=55, quality_upper=90, p=0.4),

        *_finalize_to_square(size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])


# 차량 사진류
def make_tf_car(size, mean, std):
    return A.Compose([
        # 회전
        A.OneOf([
            A.Rotate(limit=28, border_mode=0, value=255),
            A.RandomRotate90()
        ], p=0.5),
        # 확대/크롭(번호판/계기판에 집중)
        A.RandomResizedCrop(height=size, width=size, scale=(0.9, 1.0), ratio=(0.9, 1.1), interpolation=cv2.INTER_LINEAR, p=0.25),
        # 저해상/압축
        A.Downscale(scale_min=0.7, scale_max=0.9, interpolation=cv2.INTER_LINEAR, p=0.25),
        A.ImageCompression(quality_lower=50, quality_upper=90, p=0.45),
        # 흔들림/조명
        A.MotionBlur(blur_limit=3, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.22, contrast_limit=0.22, p=0.5),
        A.HueSaturationValue(hue_shift_limit=6, sat_shift_limit=12, val_shift_limit=12, p=0.3),
        A.CLAHE(clip_limit=(1, 3), p=0.25),

        *_finalize_to_square(size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ]) 


############### valid 데이터는 test 데이터처럼  ###############
def build_valid_tf(size, mean, std):
    return A.Compose([
        A.Rotate(limit=5, border_mode=0, value=255, p=0.3),
        A.ImageCompression(quality_lower=50, quality_upper=95, p=0.2),
        A.ToGray(p=0.3),
        *_finalize_to_square(size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
############################################################


# test data
def build_test_tf(size, mean, std):
    return A.Compose([
        *_finalize_to_square(size),
        # A.Normalize(mean=(0.5,)*3, std=(0.5,)*3),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])