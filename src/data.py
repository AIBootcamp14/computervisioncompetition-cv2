import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from tqdm import tqdm

try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None

from PIL import Image, ImageOps, ImageFilter


# Defaults
DEFAULT_SIZE = 672


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _guess_column(df, candidates):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in lower_map:
            return lower_map[key]
    raise ValueError(
        f"Could not find any of columns: {candidates} in {list(df.columns)}")


def _read_meta(meta_path: Path) -> Tuple[pd.DataFrame, Dict[int, str]]:
    meta = pd.read_csv(meta_path)
    id_col = _guess_column(
        meta, ["id", "label_id", "label", "class_id", "target"])
    name_col = _guess_column(meta, [
                             "name", "class", "class_name", "label_name", "labeltext", "label_name"])
    meta = meta[[id_col, name_col]].copy()
    meta.columns = ["id", "name"]
    meta["id"] = meta["id"].astype(int)
    id2name = dict(zip(meta["id"].tolist(), meta["name"].tolist()))
    return meta, id2name


def _read_train(train_csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(train_csv_path)

    if 'ID' in df.columns and 'target' in df.columns:
        df = df[['ID', 'target']].copy()
        df.columns = ["filename", "label"]
    else:
        fname_col = _guess_column(
            df, ["filename", "file", "image", "img", "path", "id"])
        label_col = _guess_column(
            df, ["label", "target", "class_id", "label_id", "y"])
        df = df[[fname_col, label_col]].copy()
        df.columns = ["filename", "label"]

    df["filename"] = df["filename"].astype(str)
    df["label"] = df["label"].astype(int)
    return df


def _exif_correct(img: Image.Image) -> Image.Image:
    try:
        return ImageOps.exif_transpose(img)
    except Exception:
        return img


def _clahe_and_denoise_pil(img: Image.Image) -> Image.Image:
    try:
        eq = ImageOps.equalize(img.convert("RGB"))
    except Exception:
        eq = img.convert("RGB")
    return eq.filter(ImageFilter.MedianFilter(size=3))


def _preprocess_image(in_path: Path, out_path: Path, size: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(in_path) as im:
        im = _exif_correct(im)
        im = im.convert("RGB")

        if cv2 is not None and np is not None:
            # Convert to OpenCV BGR
            img_cv = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
            # CLAHE on L channel in LAB
            lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            img_cv = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            # Light denoise
            try:
                img_cv = cv2.fastNlMeansDenoisingColored(
                    img_cv, None, 3, 3, 7, 21)
            except Exception:
                pass
            # Square pad
            h, w = img_cv.shape[:2]
            m = max(h, w)
            top = (m - h) // 2
            bottom = m - h - top
            left = (m - w) // 2
            right = m - w - left
            img_cv = cv2.copyMakeBorder(
                img_cv, top, bottom, left, right, cv2.BORDER_REPLICATE)
            # Resize
            img_cv = cv2.resize(img_cv, (size, size),
                                interpolation=cv2.INTER_CUBIC)
            # Convert back to PIL RGB
            im_out = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        else:
            # PIL-only fallback
            im = _clahe_and_denoise_pil(im)
            # Square pad
            w, h = im.size
            m = max(w, h)
            pad_color = (int(0.5 * 255),) * 3
            square = Image.new("RGB", (m, m), color=pad_color)
            square.paste(im, ((m - w) // 2, (m - h) // 2))
            im_out = square.resize((size, size), Image.BICUBIC)

        im_out.save(out_path, quality=95)


def prepare(project_root: Path, size: int = DEFAULT_SIZE, force_preprocess: bool = False):
    """
    - Reads meta.csv and train.csv
    - Preprocesses images from data/train and data/test into src/images/{train,test}
    - No train/val split
    Returns: (meta_df, id2name, train_df)
    """
    root = Path(project_root)
    data_dir = root / "data"
    src_dir = root / "src"
    images_out = src_dir / "images"

    meta_path = data_dir / "meta.csv"
    train_csv_path = data_dir / "train.csv"
    train_img_dir = data_dir / "train"
    test_img_dir = data_dir / "test"

    if not meta_path.exists() or not train_csv_path.exists():
        raise FileNotFoundError(
            "meta.csv or train.csv not found under data/.")

    # Read metadata
    meta_df, id2name = _read_meta(meta_path)
    train_df = _read_train(train_csv_path)

    print(f"Total images in train.csv: {len(train_df)}")
    print(f"Number of classes: {len(id2name)}")

    # Preprocess images only if needed
    out_train = images_out / "train"
    out_test = images_out / "test"

    # Check if preprocessing is needed
    need_preprocess = force_preprocess or not out_train.exists()

    if need_preprocess:
        print("Starting image preprocessing...")
        _ensure_dir(out_train)
        _ensure_dir(out_test)

        # Process all train images
        for fn in tqdm(train_df["filename"].tolist(), desc="Preprocessing train"):
            in_fp = train_img_dir / fn
            out_fp = out_train / fn

            # Skip if already exists and not forcing
            if out_fp.exists() and not force_preprocess:
                continue

            try:
                _preprocess_image(in_fp, out_fp, size)
            except Exception as e:
                print(f"[WARN] Failed preprocessing {in_fp}: {e}")

        # Process test images
        if test_img_dir.exists():
            test_files = sorted([f for f in os.listdir(test_img_dir)
                                 if (test_img_dir / f).is_file()])
            for fn in tqdm(test_files, desc="Preprocessing test"):
                in_fp = test_img_dir / fn
                out_fp = out_test / fn

                # Skip if already exists and not forcing
                if out_fp.exists() and not force_preprocess:
                    continue

                try:
                    _preprocess_image(in_fp, out_fp, size)
                except Exception as e:
                    print(f"[WARN] Failed preprocessing {in_fp}: {e}")
            print(f"Processed {len(test_files)} test images")
    else:
        print("Preprocessed images already exist. Skipping preprocessing.")
        print("Use --force to reprocess images.")

    return meta_df, id2name, train_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=str,
                        required=True, help="Path to CV_VLM root")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE,
                        help="Image size for preprocessing")
    parser.add_argument("--force", action="store_true",
                        help="Force reprocessing even if files exist")
    args = parser.parse_args()

    meta_df, id2name, train_df = prepare(
        Path(args.project_root),
        size=args.size,
        force_preprocess=args.force
    )
    print(f"Prepared. Classes: {len(id2name)}, Train images: {len(train_df)}")


if __name__ == "__main__":
    main()
