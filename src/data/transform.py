import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# set timm normalization params & resolution size
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMG_SIZE = 608  # 512, 576, 608, 640


def _finalize_to_square(size: int = IMG_SIZE):
    # fit img size square wrt longer side, pad the shorter side
    return [
        A.LongestMaxSize(max_size=size, always_apply=True),
        A.PadIfNeeded(min_height=size, min_width=size, border_mode=0, value=255, always_apply=True)
    ]

# train data : augmentation -> square화 -> normalize & tensorize
def build_train_tf_base(size: int = IMG_SIZE):
    return A.Compose([
        A.Rotate(limit=10, border_mode=0, value=255, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.GaussNoise(var_limit=(5.0, 15.0), p=0.4),
        A.ImageCompression(quality_lower=60, quality_upper=90, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.3),

        # square화 
        *_finalize_to_square(size),

        # normalize & tensorize
        # A.Normalize(mean=(0.5,)*3, std=(0.5,)*3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])


# train data (minor class): heavy augmentation & include affine -> square화 -> normalize & tensorize
def build_train_tf_heavy(size: int = IMG_SIZE):
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
        A.ImageCompression(quality_lower=40, quality_upper=85, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.2, p=0.4),

        # Affine for minor class (minor class are easily overfitted) -> minor class imgs are shown differently every epoch
        A.Affine(scale=(0.95,1.05), shear=(-4,4), translate_percent=(0.0,0.02),
                 fit_output=True, mode=0, cval=255, p=0.3),
        
        # square화      
        *_finalize_to_square(size),

        # normalize & tensorize
        # A.Normalize(mean=(0.5,)*3, std=(0.5,)*3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])


# valid/test data
def build_valid_tf(size: int = IMG_SIZE):
    return A.Compose([
        *_finalize_to_square(size),
        # A.Normalize(mean=(0.5,)*3, std=(0.5,)*3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])



# 이미지 클래스별로 증강 기법을 다르게 적용하기
# 문서류
def make_tf_doc(size: int = IMG_SIZE) -> A.Compose:
    return A.Compose([
        # 회전/기울이기
        A.Affine(rotate=(-8, 8), shear=(-3, 3), translate_percent=(0.0, 0.02),
                 fit_output=True, mode=0, cval=255, p=0.25),
        # 채도 변화
        A.HueSaturationValue(hue_shift_limit=4, sat_shift_limit=10, val_shift_limit=10, p=0.25),
        # 대비 변화
        A.CLAHE(clip_limit=(1, 3), p=0.25),
        # 저해상/압축
        A.Downscale(scale_min=0.7, scale_max=0.9, interpolation=cv2.INTER_LINEAR, p=0.2),
        A.ImageCompression(quality_lower=60, quality_upper=90, p=0.35),
        A.GaussNoise(var_limit=(5.0, 15.0), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.3),
        A.ToGray(p=0.3),

        *_finalize_to_square(size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])


# 신분증/증서류
def make_tf_idcert(size: int = IMG_SIZE) -> A.Compose:
    return A.Compose([
        A.Perspective(scale=(0.02, 0.06), pad_mode=0, pad_val=255, p=0.25),
        A.Affine(scale=(0.97, 1.03), rotate=(-6, 6), shear=(-4, 4),
                 translate_percent=(0.0, 0.02),
                 fit_output=True, mode=0, cval=255, p=0.35),
        # 가리기
        A.CoarseDropout(max_holes=2, max_height=int(size*0.12), max_width=int(size*0.12),
                        fill_value=255, p=0.2),
        # 채도 변화
        A.HueSaturationValue(hue_shift_limit=4, sat_shift_limit=12, val_shift_limit=12, p=0.25),
        # 샤픈으로 미세 텍스트 엣지 강화
        A.Sharpen(alpha=(0.05, 0.15), lightness=(0.9, 1.1), p=0.2),
        # 저해상/압축
        A.Downscale(scale_min=0.7, scale_max=0.9, interpolation=cv2.INTER_LINEAR, p=0.2),
        A.ImageCompression(quality_lower=55, quality_upper=90, p=0.4),

        *_finalize_to_square(size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])


# 차량 사진류
def make_tf_car(size: int = IMG_SIZE) -> A.Compose:
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
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])