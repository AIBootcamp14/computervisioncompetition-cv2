import cv2
import random
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from collections import OrderedDict
try:
    from skimage.restoration import richardson_lucy, unsupervised_wiener
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False


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
        A.OneOf([A.RandomRotate90(), A.Rotate(limit=10, border_mode=0, value=255)],  p=0.5),
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
        ], p=0.60),

        A.OneOf([
            A.GaussNoise(var_limit=(12.0, 28.0)),
            A.ISONoise(intensity=(0.2, 0.6)),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1)),
        ], p=0.65),
        A.MotionBlur(blur_limit=5, p=0.30),

        A.RandomBrightnessContrast(brightness_limit=0.18, contrast_limit=0.25, p=0.45),
        A.CLAHE(clip_limit=(1, 3), p=0.35),
        A.Sharpen(alpha=(0.05, 0.15), lightness=(0.9, 1.1), p=0.20),

        A.ToGray(p=0.50),
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

class FastNlMeans(A.ImageOnlyTransform):
    def __init__(self, h=7, hColor=7, templateWindowSize=7, searchWindowSize=21, p=0.35):
        super().__init__(p=p)
        self.h, self.hColor = float(h), float(hColor)
        self.tws, self.sws = int(templateWindowSize), int(searchWindowSize)

    def apply(self, img, **params):
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        den = cv2.fastNlMeansDenoisingColored(bgr, None,
                                              h=self.h, hColor=self.hColor,
                                              templateWindowSize=self.tws,
                                              searchWindowSize=self.sws)
        return cv2.cvtColor(den, cv2.COLOR_BGR2RGB)


class BilateralDenoise(A.ImageOnlyTransform):
    def __init__(self, d=5, sigmaColor=35, sigmaSpace=35, p=0.30):
        super().__init__(p=p)
        self.d = int(d); self.sc = float(sigmaColor); self.ss = float(sigmaSpace)

    def apply(self, img, **params):
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out = cv2.bilateralFilter(bgr, self.d, self.sc, self.ss)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


class UnsharpMask(A.ImageOnlyTransform):
    def __init__(self, radius=1.5, amount=0.8, threshold=0, p=0.45):
        super().__init__(p=p)
        self.radius = float(radius); self.amount = float(amount); self.threshold = int(threshold)

    def apply(self, img, **params):
        blur = cv2.GaussianBlur(img, (0, 0), self.radius)
        if self.threshold > 0:
            low_contrast_mask = np.abs(img.astype(np.int16) - blur.astype(np.int16)) < self.threshold
            sharp = cv2.addWeighted(img, 1 + self.amount, blur, -self.amount, 0)
            sharp = np.where(low_contrast_mask, img, sharp)
        else:
            sharp = cv2.addWeighted(img, 1 + self.amount, blur, -self.amount, 0)
        return np.clip(sharp, 0, 255).astype(np.uint8)


def _variance_of_laplacian(gray_u8):
    return cv2.Laplacian(gray_u8, cv2.CV_64F).var()


class ConditionalDeblur(A.ImageOnlyTransform):
    def __init__(self,
                 blur_thresh=110.0, 
                 use_rl=True, max_iter=15,
                 wiener=True, p=0.40):
        super().__init__(p=p)
        self.blur_thresh = float(blur_thresh)
        self.use_rl = bool(use_rl)
        self.max_iter = int(max_iter)
        self.use_wiener = bool(wiener)


    def _simple_motion_psf(self, length=7, angle=0.0):

        psf = np.zeros((length, length), dtype=np.float32)
        psf[length//2, :] = 1.0
        psf /= psf.sum()
        M = cv2.getRotationMatrix2D((length/2, length/2), angle, 1.0)
        psf = cv2.warpAffine(psf, M, (length, length))
        psf = psf / (psf.sum() + 1e-8)
        return psf


    def apply(self, img, **params):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        score = _variance_of_laplacian(gray)
        if score >= self.blur_thresh:

            us = UnsharpMask(radius=1.2, amount=0.6, threshold=0, p=1.0)
            return us.apply(img)


        if _HAS_SKIMAGE and (self.use_rl or self.use_wiener):
            psf = self._simple_motion_psf(length=9, angle=0.0)
            img_f = img.astype(np.float32) / 255.0
            if self.use_wiener:
                try:
                    chans = []
                    for c in cv2.split(img_f):
                        rec, _ = unsupervised_wiener(c, psf)
                        chans.append(np.clip(rec, 0, 1))
                    rec = cv2.merge(chans)
                    out = (rec * 255.0).astype(np.uint8)
                    return out
                except Exception:
                    pass
            if self.use_rl:
                chans = []
                for c in cv2.split(img_f):
                    rec = richardson_lucy(c, psf, num_iter=self.max_iter, clip=True)
                    chans.append(np.clip(rec, 0, 1))
                rec = cv2.merge(chans)
                out = (rec * 255.0).astype(np.uint8)
                return out

        us = UnsharpMask(radius=1.8, amount=0.9, threshold=0, p=1.0)
        return us.apply(img)


def build_test_tf(size, mean, std, interpolation=cv2.INTER_LINEAR):
    return A.Compose([
        *_finalize_to_square(size, interpolation),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])


def build_test_tf_doc_deskew(size, mean, std, interpolation=cv2.INTER_LINEAR):
    return A.Compose([
        A.Affine(rotate=(-10, 10), shear=(-4, 4), translate_percent=(0.0, 0.01),
                 fit_output=True, mode=0, cval=255, p=0.9),
        *_finalize_to_square(size, interpolation),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])


def build_test_tf_doc_restore_bal(size, mean, std, interpolation=cv2.INTER_LINEAR):
    return A.Compose([
        A.Affine(rotate=(-10, 10), shear=(-4, 4), translate_percent=(0.0, 0.015),
                 fit_output=True, mode=0, cval=255, p=0.65),
        A.OneOf([
            FastNlMeans(h=6, hColor=6, templateWindowSize=7, searchWindowSize=21, p=1.0),
            BilateralDenoise(d=5, sigmaColor=30, sigmaSpace=30, p=1.0),
        ], p=0.60),
        A.OneOf([
            A.CLAHE(clip_limit=(1, 3)),
            A.Equalize(mode="cv", by_channels=True),
        ], p=0.55),
        A.RandomBrightnessContrast(brightness_limit=0.07, contrast_limit=0.12, p=0.55),
        ConditionalDeblur(blur_thresh=120.0, use_rl=True, max_iter=12, wiener=True, p=0.45),
        UnsharpMask(radius=1.5, amount=0.75, threshold=0, p=0.45),
        A.ToGray(p=0.45),
        *_finalize_to_square(size, interpolation),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])


def build_test_tf_doc_restore_max(size, mean, std, interpolation=cv2.INTER_LINEAR):
    return A.Compose([
        A.Affine(rotate=(-12, 12), shear=(-5, 5), translate_percent=(0.0, 0.02),
                 fit_output=True, mode=0, cval=255, p=0.75),
        A.OneOf([
            FastNlMeans(h=7.5, hColor=7.5, templateWindowSize=7, searchWindowSize=21, p=1.0),
            BilateralDenoise(d=7, sigmaColor=38, sigmaSpace=38, p=1.0),
        ], p=0.75),
        A.CLAHE(clip_limit=(2, 3), p=0.65),
        A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.14, p=0.60),
        ConditionalDeblur(blur_thresh=140.0, use_rl=True, max_iter=16, wiener=True, p=0.55),
        UnsharpMask(radius=1.8, amount=0.9, threshold=0, p=0.60),
        A.ToGray(p=0.50),
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