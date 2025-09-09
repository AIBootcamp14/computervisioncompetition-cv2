from torch.utils.data import Dataset
from collections import OrderedDict
import cv2, random, os
import numpy as np

class ImgDataset(Dataset):
    def __init__(self, df, tf_base, tf_heavy=None,
                minor_classes=None, p_heavy_minor=0.7, use_heavy_for_all=False,
                base_dir=None, class_tf_map: dict[int, object] | None = None):
        self.df = df.reset_index(drop=True)
        self.tf_base = tf_base
        self.tf_heavy = tf_heavy
        self.minor = set(minor_classes or [])
        self.p_minor = p_heavy_minor
        self.use_heavy_for_all = use_heavy_for_all
        self.base_dir = base_dir
        self.class_tf_map = class_tf_map or {}

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, i):
        base = os.path.abspath(self.base_dir or "data/train")
        raw = str(self.df.loc[i, 'ID'])

        path = os.path.abspath(raw) if os.path.isabs(raw) else os.path.abspath(os.path.join(base, raw))
        if os.path.commonpath([base, path]) != base:
            raise ValueError(f"Out-of-root path detected: {path}")

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"No image found: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.ascontiguousarray(img.astype(np.uint8))

        has_target = 'target' in self.df.columns
        label = int(self.df.loc[i, 'target']) if has_target else None

        tf = self.tf_base
        if has_target and self.class_tf_map:
            tf = self.class_tf_map.get(label, tf)
        elif self.tf_heavy is not None and has_target:
            if (label in self.minor and random.random() < self.p_minor) or (self.use_heavy_for_all and random.random() < 0.5):
                tf = self.tf_heavy

        img = tf(image=img)['image']
        return (img, label) if has_target else (img, raw)


class CachedDataset(Dataset):
    def __init__(self, base_ds, max_cache_size=None):
        self.base_ds = base_ds
        self.max_cache_size = max_cache_size or len(base_ds)
        self._cache = OrderedDict()

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, i):
        if i in self._cache:
            val = self._cache.pop(i)
            self._cache[i] = val
            return val
        val = self.base_ds[i]
        while len(self._cache) >= self.max_cache_size:
            self._cache.popitem(last=False)
        self._cache[i] = val
        return val