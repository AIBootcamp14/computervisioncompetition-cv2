from torch.utils.data import Dataset
import cv2, random, os

class ImgDataset(Dataset):
    """
    df: ['ID'] (+ optional 'target')
    base_dir: 'data/train' or 'data/test'
    """

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
        raw = str(self.df.loc[i, 'ID'])
        path = raw if (os.path.isabs(raw) or any(sep in raw for sep in ("/","\\"))) \
               else os.path.join(self.base_dir or "data/train", raw)

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        has_target = 'target' in self.df.columns
        label = int(self.df.loc[i, 'target']) if has_target else None

        use_heavy = False
        if self.tf_heavy is not None and has_target:
            if label in self.minor and random.random() < self.p_minor:
                use_heavy = True
            elif self.use_heavy_for_all and random.random() < 0.5:
                use_heavy = True

        tf = self.tf_base
        if has_target and self.class_tf_map:
            tf = self.class_tf_map.get(label, tf)
        elif self.tf_heavy is not None:
            if has_target and (
                (label in self.minor and random.random() < self.p_minor) or
                (self.use_heavy_for_all and random.random() < 0.5)
            ):
                tf = self.tf_heavy

        img = tf(image=img)['image']
        return (img, label) if has_target else (img, raw)
