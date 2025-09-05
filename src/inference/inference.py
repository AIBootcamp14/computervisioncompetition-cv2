import os
import json
import argparse
import warnings
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import timm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from timm.data import resolve_model_data_config
from src.data.transform import build_test_tf
from src.train.train import set_seed


def _resolve_img_size_from_model(model, user_opt):
    _, in_h, in_w = resolve_model_data_config(model)['input_size']
    if user_opt.isdigit():
        forced = int(user_opt)
        warnings.warn(f"Infer stage: Override img_size to {forced} (model default: {in_h}x{in_w})", UserWarning)
        return forced
    if in_h != in_w:
        if user_opt == "auto-long":
            size = max(in_h, in_w)
            warnings.warn(f"Infer stage: Rectangular default ({in_h}x{in_w}). Using longer side = {size}.", UserWarning)
            return size
        else:
            warnings.warn(f"Infer stage: Rectangular default ({in_h}x{in_w}). Using height = {in_h}.", UserWarning)
            return in_h
    return in_h


def _tta_stream(x, enable):
    # TTA 텐서를 하나씩 생성해서 사용 (OOM 방지)
    yield x
    if enable:
        yield torch.flip(x, dims=[3])              # hflip
        yield torch.flip(x, dims=[2])              # vflip
        yield torch.rot90(x, k=1, dims=[2, 3])     # 90
        yield torch.rot90(x, k=2, dims=[2, 3])     # 180
        yield torch.rot90(x, k=3, dims=[2, 3])     # 270


class TestImageDataset(Dataset):
    """
    Infer target of imgs under test_dir. 
    returns: (tensor, filename)
    """
    def __init__(self, test_dir, transform=None, exts=(".jpg", ".png", ".jpeg")):
        self.test_dir = test_dir
        self.transform = transform
        self.files = sorted([
            f for f in os.listdir(test_dir)
            if f.lower().endswith(exts)
        ])
        if len(self.files) == 0:
            raise FileNotFoundError(f"No image files found in: {test_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        fpath = os.path.join(self.test_dir, fname)
        img = Image.open(fpath).convert("RGB")
        img = np.array(img)
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img, fname


def load_fold_models(summary_path, device):
    """
    model/{arch}_{img_size_opt}/train_summary.json 을 열어 fold별 ckpt를 로드합니다.
    """
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Summary not found: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        fold_logs = json.load(f)

    models = []
    for item in fold_logs:
        ckpt_path = item["ckpt"]
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        arch = ckpt["arch"]
        num_classes = ckpt["num_classes"]

        model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.eval()
        models.append(model)
        print(f"Loaded: {os.path.basename(ckpt_path)} | arch={arch} | num_classes={num_classes}")

    if not models:
        raise RuntimeError("No models loaded from summary.")
    return models


@torch.no_grad()
def infer_ensemble(models, loader, device, use_tta=False, return_proba=False):
    ids, preds, all_proba = [], [], []

    for xb, fnames in tqdm(loader, desc="Infer", leave=False):
        xb = xb.to(device, non_blocking=True)

        prob_sum = None  # (B, C)
        for m in models:
            # fold별로 GPU에 올렸다가 바로 내리기
            m = m.to(device)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                prob_sum_model = None
                tta_count = 0

                for v in _tta_stream(xb, enable=use_tta):
                    logits = m(v)
                    probs = torch.softmax(logits, dim=1)   # (B, C)
                    prob_sum_model = probs if prob_sum_model is None else (prob_sum_model + probs)
                    tta_count += 1

                prob_sum_model = prob_sum_model / tta_count  # TTA 평균

            prob_sum = prob_sum_model if prob_sum is None else (prob_sum + prob_sum_model)

            # fold 하나 끝났으면 GPU에서 캐시 비우기
            m = m.to("cpu")
            torch.cuda.empty_cache()

        prob_avg = (prob_sum / len(models)).float().cpu().numpy()  # fold 평균
        pred = prob_avg.argmax(axis=1)

        ids.extend(fnames)
        preds.extend(pred.tolist())
        if return_proba:
            all_proba.append(prob_avg)

    df = pd.DataFrame({"ID": ids, "target": preds})
    if return_proba:
        proba = np.concatenate(all_proba, axis=0)
        proba_cols = [f"prob_{i}" for i in range(proba.shape[1])]
        df = pd.concat([df, pd.DataFrame(proba, columns=proba_cols)], axis=1)
    return df


def run_inference():
    set_seed(42)

    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", type=str, required=True)
    ap.add_argument("--test_dir", type=str, default="data/test")
    ap.add_argument("--summary_path", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default="output")
    ap.add_argument("--img_size", type=str, default="auto")
    ap.add_argument("--infer_batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--save_proba", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    arch_dir = f"{args.arch}_{args.img_size}"
    summary_path = args.summary_path or os.path.join("model", arch_dir, "train_summary.json")

    models = load_fold_models(summary_path, device)
    data_cfg = resolve_model_data_config(models[0])
    mean = tuple(map(float, data_cfg["mean"]))
    std  = tuple(map(float, data_cfg["std"]))
    img_size = _resolve_img_size_from_model(models[0], args.img_size)

    ds = TestImageDataset(args.test_dir, transform=build_test_tf(img_size, mean, std))
    dl = DataLoader(ds, batch_size=args.infer_batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    df = infer_ensemble(models, dl, device, use_tta=args.tta, return_proba=args.save_proba)

    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, f"{arch_dir}.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

if __name__ == "__main__":
    run_inference()
