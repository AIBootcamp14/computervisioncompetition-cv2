
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
data_08.py — robust document preprocessor (orientation + OCR-based mirror fix)

Major features
- White-border cropping to avoid background dominating orientation
- Quadrant selection (0/90/180/270) by text-gradient heuristic
- Deskew up to ±max_skew_deg (default 35°) using Hough lines
- Optional Tesseract-based mirror detection & fix (左右 flip)
- Letterbox pad & resize, output PNG/JPG
- Single set or train+test ("both") modes
- Multiprocessing-safe worker (top-level function)

Requirements
- OpenCV (cv2), numpy, Pillow, pandas
- (For mirror fix) pytesseract + system tesseract binary with selected language data

Usage (test only):
  python src/data_08.py --mode single \
    --csv data/sample_submission.csv \
    --input_dir data/test \
    --out_dir src/images/test \
    --size 672 --format png \
    --auto_orient --max_skew_deg 35 --always_rotate_docs \
    --try_mirror --ocr_lang kor+eng --mirror_min_gain 5 \
    --num_workers 8 --force --quiet

Usage (train + test):
  python src/data_08.py --mode both \
    --train_csv data/train.csv --train_input_dir data/train --train_out_dir src/images/train \
    --test_csv data/sample_submission.csv --test_input_dir data/test --test_out_dir src/images/test \
    --size 672 --format png \
    --auto_orient --max_skew_deg 35 --always_rotate_docs \
    --try_mirror --ocr_lang kor+eng --mirror_min_gain 5 \
    --num_workers 8 --force --quiet
"""
from __future__ import annotations

import os
import argparse
from typing import List, Optional, Tuple

import numpy as np
import cv2
from PIL import Image
import pandas as pd
from tqdm import tqdm

# -------------------------- CSV / IO helpers --------------------------


def ensure_dir(p: str):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def find_col(cands: List[str], cols: List[str]) -> Optional[str]:
    lowers = [c.lower() for c in cols]
    for c in cands:
        if c in cols:
            return c
        if c.lower() in lowers:
            return cols[lowers.index(c.lower())]
    return None


def load_ids(csv_path: str) -> List[str]:
    df = pd.read_csv(csv_path)
    id_col = find_col(["ID", "id", "image", "img", "filename",
                      "file", "filepath", "path", "image_id"], list(df.columns))
    if id_col is None:
        raise ValueError(
            f"{csv_path} must have an image id column (e.g., ID).")
    return [str(x) for x in df[id_col].tolist()]


def build_records(ids: List[str], input_dir: str, out_dir: str, fmt: str, force: bool) -> List[Tuple[str, str]]:
    recs: List[Tuple[str, str]] = []
    for fid in ids:
        stem, ext = os.path.splitext(str(fid))
        # resolve source
        src = None
        p0 = os.path.join(input_dir, str(fid))
        if os.path.exists(p0):
            src = p0
        else:
            for e in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".JPG", ".PNG", ".JPEG"]:
                p = os.path.join(input_dir, stem + e)
                if os.path.exists(p):
                    src = p
                    break
        if src is None:
            # still try to process to count + warn inside worker
            src = p0
        # resolve dest
        ext_out = (ext[1:] if (fmt == "same" and ext) else fmt).lower()
        if ext_out not in ("png", "jpg", "jpeg"):
            ext_out = "png"
        dst = os.path.join(out_dir, stem + "." + ext_out)
        if (not force) and os.path.exists(dst):
            continue
        recs.append((src, dst))
    return recs

# -------------------------- Orientation core --------------------------


def crop_white_border(img_bgr: np.ndarray, pad: int = 8) -> np.ndarray:
    """Keep largest contour around content. Works even with wide margins."""
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    b = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY_INV, 41, 15)
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones(
        (3, 3), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img_bgr
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    x = max(0, x-pad)
    y = max(0, y-pad)
    return img_bgr[y:y+h+2*pad, x:x+w+2*pad]


def best_quadrant_by_gradient(img_bgr: np.ndarray) -> np.ndarray:
    """Choose orientation among (0,90,180,270) by emphasizing text strokes."""
    roi = crop_white_border(img_bgr, pad=8)
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)

    def score(gray: np.ndarray) -> float:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        # horizontal text -> vertical gradient strong -> score larger
        return float(np.sum(np.abs(gy)) - np.sum(np.abs(gx)))

    cands_roi = [
        roi,
        cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(roi, cv2.ROTATE_180),
        cv2.rotate(roi, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]
    scores = [score(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)) for c in cands_roi]
    k = int(np.argmax(scores))
    # return same rotation applied to full image
    return [
        img_bgr,
        cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(img_bgr, cv2.ROTATE_180),
        cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ][k]


def estimate_skew_hough(gray: np.ndarray, limit_deg: float = 35.0) -> float:
    edges = cv2.Canny(gray, 60, 180)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=120)
    if lines is None:
        return 0.0
    angs = []
    for rho, theta in lines[:, 0, :]:
        deg = (theta*180/np.pi) - 90.0     # 0: horizontal
        deg = ((deg + 90 + 180) % 180) - 90
        if abs(deg) <= limit_deg:
            angs.append(deg)
    if not angs:
        return 0.0
    return float(np.median(angs))


def auto_upright_doc(img_bgr: np.ndarray, limit_deg: float = 35.0) -> np.ndarray:
    q = best_quadrant_by_gradient(img_bgr)
    g = cv2.cvtColor(q, cv2.COLOR_BGR2GRAY)
    ang = estimate_skew_hough(g, limit_deg=limit_deg)
    if abs(ang) < 0.5:
        return q
    h, w = q.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), ang, 1.0)
    return cv2.warpAffine(q, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def looks_document_like(img_bgr: np.ndarray) -> bool:
    roi = crop_white_border(img_bgr, pad=8)
    h, w = img_bgr.shape[:2]
    area_ratio = (roi.shape[0]*roi.shape[1]) / max(1, h*w)
    return area_ratio >= 0.15

# -------------------------- OCR-based mirror fix --------------------------


def assert_tesseract_available():
    import pytesseract
    try:
        _ = pytesseract.get_tesseract_version()
    except Exception as e:
        raise RuntimeError(
            "Tesseract is required for --try_mirror.\n"
            "Install it first (e.g., apt-get install -y tesseract-ocr tesseract-ocr-kor)\n"
            f"Underlying error: {e}"
        )


def ocr_confidence_bgr(img_bgr: np.ndarray, lang: str = "kor+eng") -> float:
    """Return median OCR confidence using Tesseract. -1 if no text boxes."""
    import pytesseract
    from pytesseract import Output
    cfg = "--oem 1 --psm 6"
    data = pytesseract.image_to_data(
        cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
        lang=lang, config=cfg, output_type=Output.DICT
    )
    confs = [int(c) for c in data.get("conf", [])
             if str(c) not in ("-1", "-1.0")]
    return float(np.median(confs)) if confs else -1.0


def maybe_fix_mirror(img_bgr: np.ndarray, lang: str = "kor+eng", min_gain: float = 5.0) -> Tuple[np.ndarray, bool, float, float]:
    """Flip horizontally if OCR conf improves by >= min_gain."""
    c0 = ocr_confidence_bgr(img_bgr, lang=lang)
    flipped = cv2.flip(img_bgr, 1)
    c1 = ocr_confidence_bgr(flipped, lang=lang)
    if c1 > c0 + float(min_gain):
        return flipped, True, c0, c1
    return img_bgr, False, c0, c1

# -------------------------- Resize & save --------------------------


def letterbox_square(img_bgr: np.ndarray, size: int = 672, pad_color=(128, 128, 128)) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h*scale)), int(round(w*scale))
    resized = cv2.resize(
        img_bgr, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    top = (size - nh) // 2
    left = (size - nw) // 2
    out = np.full((size, size, 3), pad_color, dtype=np.uint8)
    out[top:top+nh, left:left+nw] = resized
    return out


def save_image(img_bgr: np.ndarray, dst_path: str):
    ensure_dir(os.path.dirname(dst_path))
    ext = os.path.splitext(dst_path)[1].lower()
    if ext == ".png":
        cv2.imwrite(dst_path, img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    elif ext in (".jpg", ".jpeg"):
        cv2.imwrite(dst_path, img_bgr, [
                    cv2.IMWRITE_JPEG_QUALITY, 95, cv2.IMWRITE_JPEG_OPTIMIZE, 1])
    else:
        cv2.imwrite(dst_path, img_bgr)

# -------------------------- Worker (top-level for multiprocessing) --------------------------


def worker_task(task):
    (src, dst, size, do_orient, limit_deg, always_rotate_docs,
     try_mirror, ocr_lang, mirror_min_gain) = task
    img = cv2.imread(src, cv2.IMREAD_COLOR)
    if img is None:
        return f"[warn] cannot read: {src}"

    # orientation
    if do_orient and (always_rotate_docs or looks_document_like(img)):
        img = auto_upright_doc(img, limit_deg=limit_deg)

    # mirror (requires tesseract)
    if try_mirror:
        img, flipped, c0, c1 = maybe_fix_mirror(
            img, lang=ocr_lang, min_gain=mirror_min_gain)
        # keep silent; return brief message only when flipped
        if flipped:
            return f"[mirror] {os.path.basename(src)} conf {c0:.1f}->{c1:.1f}"

    # resize & save
    img = letterbox_square(img, size=size, pad_color=(128, 128, 128))
    save_image(img, dst)
    return None

# -------------------------- Runner --------------------------


def run_batch(name: str, ids: List[str], input_dir: str, out_dir: str,
              size: int, fmt: str, num_workers: int, force: bool,
              do_orient: bool, limit_deg: float, always_rotate_docs: bool,
              try_mirror: bool, ocr_lang: str, mirror_min_gain: float,
              quiet: bool):
    recs = build_records(ids, input_dir, out_dir, fmt, force)
    total = len(recs)
    if not quiet:
        print(f"[{name}] total to process: {total}")

    tasks = [(src, dst, size, do_orient, limit_deg, always_rotate_docs,
              try_mirror, ocr_lang, mirror_min_gain) for (src, dst) in recs]

    if try_mirror:
        # ensure tesseract is available once up front
        assert_tesseract_available()

    if num_workers > 1:
        from multiprocessing import Pool
        with Pool(processes=num_workers) as pool:
            for msg in tqdm(pool.imap_unordered(worker_task, tasks, chunksize=32),
                            total=total, desc=f"[{name}] Processing", ncols=100):
                if msg and (not quiet):
                    print(msg)
    else:
        for t in tqdm(tasks, total=total, desc=f"[{name}] Processing", ncols=100):
            msg = worker_task(t)
            if msg and (not quiet):
                print(msg)

# -------------------------- CLI --------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "both"], required=True)

    # single
    ap.add_argument("--csv")
    ap.add_argument("--input_dir")
    ap.add_argument("--out_dir")

    # both
    ap.add_argument("--train_csv")
    ap.add_argument("--train_input_dir")
    ap.add_argument("--train_out_dir")
    ap.add_argument("--test_csv")
    ap.add_argument("--test_input_dir")
    ap.add_argument("--test_out_dir")

    # options
    ap.add_argument("--size", type=int, default=672)
    ap.add_argument(
        "--format", choices=["png", "jpg", "jpeg", "same"], default="png")
    ap.add_argument("--auto_orient", action="store_true")
    ap.add_argument("--max_skew_deg", type=float, default=35.0)
    ap.add_argument("--always_rotate_docs", action="store_true",
                    help="ignore gate and always run orientation if --auto_orient")
    ap.add_argument("--try_mirror", action="store_true",
                    help="compare OCR confidence and flip horizontally if improved")
    ap.add_argument("--ocr_lang", default="kor+eng")
    ap.add_argument("--mirror_min_gain", type=float, default=5.0)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.mode == "single":
        if not all([args.csv, args.input_dir, args.out_dir]):
            raise ValueError(
                "--csv, --input_dir, --out_dir are required for --mode single")
        ids = load_ids(args.csv)
        run_batch("SINGLE", ids, args.input_dir, args.out_dir,
                  size=args.size, fmt=args.format, num_workers=args.num_workers, force=args.force,
                  do_orient=args.auto_orient, limit_deg=args.max_skew_deg, always_rotate_docs=args.always_rotate_docs,
                  try_mirror=args.try_mirror, ocr_lang=args.ocr_lang, mirror_min_gain=args.mirror_min_gain,
                  quiet=args.quiet)

    else:  # both
        need = [args.train_csv, args.train_input_dir, args.train_out_dir,
                args.test_csv, args.test_input_dir, args.test_out_dir]
        if not all(need):
            raise ValueError(
                "--train_* and --test_* are required for --mode both")
        train_ids = load_ids(args.train_csv)
        test_ids = load_ids(args.test_csv)
        run_batch("TRAIN", train_ids, args.train_input_dir, args.train_out_dir,
                  size=args.size, fmt=args.format, num_workers=args.num_workers, force=args.force,
                  do_orient=args.auto_orient, limit_deg=args.max_skew_deg, always_rotate_docs=args.always_rotate_docs,
                  try_mirror=args.try_mirror, ocr_lang=args.ocr_lang, mirror_min_gain=args.mirror_min_gain,
                  quiet=args.quiet)
        run_batch("TEST", test_ids, args.test_input_dir, args.test_out_dir,
                  size=args.size, fmt=args.format, num_workers=args.num_workers, force=args.force,
                  do_orient=args.auto_orient, limit_deg=args.max_skew_deg, always_rotate_docs=args.always_rotate_docs,
                  try_mirror=args.try_mirror, ocr_lang=args.ocr_lang, mirror_min_gain=args.mirror_min_gain,
                  quiet=args.quiet)


if __name__ == "__main__":
    main()
