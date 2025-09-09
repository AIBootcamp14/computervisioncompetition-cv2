import cv2
import random
import torch
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from collections import OrderedDict

INTER_AUTO = "auto"


class LongestMaxSizeAuto(A.ImageOnlyTransform):
    def __init__(self, max_size, always_apply=True, p=1.0):
        super().__init__(always_apply=always_apply, p=p)
        self.max_size = int(max_size)

    def apply(self, img, **params):
        h, w = img.shape[:2]
        scale = self.max_size / float(max(h, w))
        if np.isclose(scale, 1.0, atol=1e-3):
            interp = cv2.INTER_LINEAR
        elif scale < 1.0:
            interp = cv2.INTER_AREA
        else:
            interp = cv2.INTER_CUBIC
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        return cv2.resize(img, (new_w, new_h), interpolation=interp)


def _finalize_to_square(size, interpolation=cv2.INTER_LINEAR):
    resize_tf = ([LongestMaxSizeAuto(size)] if interpolation == INTER_AUTO else [A.LongestMaxSize(max_size=size, interpolation=interpolation, always_apply=True)])
    return resize_tf + [A.PadIfNeeded(min_height=size, min_width=size, border_mode=0, value=255, always_apply=True)]


def build_train_tf_base(size, mean, std, interpolation=cv2.INTER_LINEAR):
    return A.Compose([
        A.OneOf([A.RandomRotate90(), A.Rotate(limit=10, border_mode=0, value=255, p=0.5)]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.GaussNoise(var_limit=(5.0, 15.0), p=0.4),
        A.ImageCompression(quality_lower=30, quality_upper=90, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.3),

        *_finalize_to_square(size, interpolation),

        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])


def build_train_tf_heavy(size, mean, std, interpolation=cv2.INTER_LINEAR):
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

        A.Affine(scale=(0.95,1.05), shear=(-4,4), translate_percent=(0.0,0.02),
                 fit_output=True, mode=0, cval=255, p=0.3),

        *_finalize_to_square(size, interpolation),

        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])


class ProbTransform:
    def __init__(self, base_tf, heavy_tf, p=0.7):
        self.base_tf = base_tf
        self.heavy_tf = heavy_tf
        self.p = float(p)

    def __call__(self, **kwargs):
        if random.random() < self.p:
            return self.heavy_tf(**kwargs)
        return self.base_tf(**kwargs)


class OverlayFromPool(A.ImageOnlyTransform):
    def __init__(self, paths, alpha=(0.06, 0.14), angle_set=(0, 90, 180, 270),
                 scale=(0.9, 1.1), self_or_pool_prob=0.3, p=0.10, max_cache=256):
        super().__init__(p=p)
        self.paths = list(paths or [])
        self.alpha = alpha
        self.angle_set = tuple(angle_set)
        self.scale = scale
        self.self_or_pool_prob = float(self_or_pool_prob)
        self.max_cache = int(max_cache)
        self._cache = OrderedDict()

    def _to_rgb_u8(self, arr):
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)
        return np.ascontiguousarray(arr)

    def _read_rgb_cached(self, path, shape_wh=None):
        if path in self._cache:
            img = self._cache.pop(path)
            self._cache[path] = img
        else:
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if bgr is None:
                return None
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = self._to_rgb_u8(rgb)
            while len(self._cache) >= self.max_cache:
                self._cache.popitem(last=False)
            self._cache[path] = rgb
            img = rgb
        if shape_wh is not None:
            w, h = shape_wh
            if img.shape[0] != h or img.shape[1] != w:
                interp = cv2.INTER_AREA if (h < img.shape[0] or w < img.shape[1]) else cv2.INTER_CUBIC
                img = cv2.resize(img, (w, h), interpolation=interp)
                img = self._to_rgb_u8(img)
        return img

    def apply(self, img, **params):
        h, w = img.shape[:2]
        base = self._to_rgb_u8(img)

        use_self = (not self.paths) or (random.random() < self.self_or_pool_prob)
        if use_self:
            over = base.copy()
        else:
            pth = random.choice(self.paths)
            over = self._read_rgb_cached(pth, shape_wh=(w, h))
            if over is None:
                over = base.copy()

        angle = random.choice(self.angle_set)
        sc = np.random.uniform(*self.scale)
        M = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, sc)
        over = cv2.warpAffine(
            over, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )
        over = self._to_rgb_u8(over)

        a = np.random.uniform(*self.alpha)
        mixed = cv2.addWeighted(base, 1.0, over, a, 0.0)
        return mixed


def make_tf_doc(size, mean, std, interpolation=cv2.INTER_LINEAR, overlay: A.BasicTransform | None = None):
    ops = [
        A.OneOf([
            A.Affine(rotate=(-15, 15), shear=(-5, 5),
                     translate_percent=(0.0, 0.03),
                     fit_output=True, mode=0, cval=255),
            A.RandomRotate90()
        ], p=0.45),

        A.HueSaturationValue(hue_shift_limit=4, sat_shift_limit=10, val_shift_limit=10, p=0.25),
        A.CLAHE(clip_limit=(1, 3), p=0.3),

        A.OneOf([
            A.Downscale(scale_min=0.7, scale_max=0.9, interpolation=cv2.INTER_LINEAR),
            A.ImageCompression(quality_lower=20, quality_upper=85),
        ], p=0.40),

        A.OneOf([
            A.GaussNoise(var_limit=(6.0, 18.0)),
            A.ISONoise(intensity=(0.1, 0.4)),
        ], p=0.45),
        A.MotionBlur(blur_limit=3, p=0.20),

        A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.35),
        A.ToGray(p=0.35),
        A.InvertImg(p=0.15),
    ]
    if overlay is not None:
        ops.append(overlay)

    ops += [
        *_finalize_to_square(size, interpolation),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ]
    return A.Compose(ops)


def make_tf_doc_heavy(size, mean, std, interpolation=cv2.INTER_LINEAR, overlay: A.BasicTransform | None = None):
    ops = [
        A.OneOf([
            A.Affine(rotate=(-22, 22), shear=(-7, 7),
                     translate_percent=(0.0, 0.03),
                     fit_output=True, mode=0, cval=255),
            A.RandomRotate90()
        ], p=0.65),

        A.OneOf([
            A.Downscale(scale_min=0.6, scale_max=0.85, interpolation=cv2.INTER_LINEAR),
            A.ImageCompression(quality_lower=18, quality_upper=60),
        ], p=0.65),

        A.OneOf([
            A.GaussNoise(var_limit=(12.0, 28.0)),
            A.ISONoise(intensity=(0.2, 0.6)),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1)),
        ], p=0.60),
        A.MotionBlur(blur_limit=5, p=0.30),

        A.RandomBrightnessContrast(brightness_limit=0.18, contrast_limit=0.25, p=0.45),
        A.CLAHE(clip_limit=(1, 3), p=0.35),
        A.Sharpen(alpha=(0.05, 0.15), lightness=(0.9, 1.1), p=0.20),

        A.ToGray(p=0.55),
        A.InvertImg(p=0.18),
    ]
    if overlay is not None:
        ops.append(overlay)

    ops += [
        *_finalize_to_square(size, interpolation),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ]
    return A.Compose(ops)


def make_tf_idcert(size, mean, std, interpolation=cv2.INTER_LINEAR):
    return A.Compose([
        A.Perspective(scale=(0.02, 0.06), pad_mode=0, pad_val=255, p=0.30),
        A.Affine(scale=(0.97, 1.03), rotate=(-8, 8), shear=(-5, 5),
                 translate_percent=(0.0, 0.02),
                 fit_output=True, mode=0, cval=255, p=0.45),

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.20),

        A.CoarseDropout(max_holes=3, max_height=int(size*0.12), max_width=int(size*0.12),
                        fill_value=255, p=0.20),

        A.HueSaturationValue(hue_shift_limit=4, sat_shift_limit=12, val_shift_limit=12, p=0.25),
        A.Sharpen(alpha=(0.05, 0.15), lightness=(0.9, 1.1), p=0.20),

        A.Downscale(scale_min=0.7, scale_max=0.9, interpolation=cv2.INTER_LINEAR, p=0.25),
        A.ImageCompression(quality_lower=50, quality_upper=90, p=0.45),
        A.MotionBlur(blur_limit=3, p=0.20),

        *_finalize_to_square(size, interpolation),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])


def make_tf_car(size, mean, std, interpolation=cv2.INTER_LINEAR):
    return A.Compose([
        A.OneOf([
            A.Rotate(limit=30, border_mode=0, value=255),
            A.RandomRotate90()
        ], p=0.55),

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.35),

        A.RandomResizedCrop(height=size, width=size, scale=(0.85, 1.0), ratio=(0.85, 1.15),
                            interpolation=cv2.INTER_LINEAR, p=0.30),

        A.Downscale(scale_min=0.7, scale_max=0.9, interpolation=cv2.INTER_LINEAR, p=0.30),
        A.ImageCompression(quality_lower=45, quality_upper=90, p=0.50),

        A.MotionBlur(blur_limit=5, p=0.35),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.55),
        A.HueSaturationValue(hue_shift_limit=6, sat_shift_limit=14, val_shift_limit=14, p=0.35),
        A.CLAHE(clip_limit=(1, 3), p=0.25),

        *_finalize_to_square(size, interpolation),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])


def build_test_tf(size, mean, std, interpolation=cv2.INTER_LINEAR):
    return A.Compose([
        *_finalize_to_square(size, interpolation),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])


def build_valid_tf_robust(size, mean, std, interpolation=cv2.INTER_LINEAR):
    return A.Compose([
        A.OneOf([A.Rotate(limit=20, border_mode=0, value=255), A.RandomRotate90()], p=0.4),
        A.Downscale(scale_min=0.7, scale_max=0.9, interpolation=cv2.INTER_LINEAR, p=0.4),
        A.ImageCompression(quality_lower=25, quality_upper=70, p=0.4),
        A.GaussNoise(var_limit=(10.0, 25.0), p=0.4),
        A.MotionBlur(blur_limit=5, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.25, p=0.4),
        A.ToGray(p=0.4),
        A.InvertImg(p=0.2),
        *_finalize_to_square(size, interpolation),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])