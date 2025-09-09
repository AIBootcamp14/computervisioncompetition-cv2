import os
import json
import argparse
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import timm
import torch
from torch.utils.data import Dataset, DataLoader
from timm.data import resolve_model_data_config
from src.data.transform import INTER_AUTO, build_test_tf
from src.train.train import set_seed

try:
    import wandb
except Exception:
    wandb = None
    

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
    yield x
    if enable:
        yield torch.flip(x, dims=[3])
        yield torch.flip(x, dims=[2])
        yield torch.rot90(x, k=1, dims=[2, 3]) 
        yield torch.rot90(x, k=2, dims=[2, 3]) 
        yield torch.rot90(x, k=3, dims=[2, 3])


class TestImageDataset(Dataset):
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
        with Image.open(fpath) as im:
            img = np.array(im.convert("RGB"))
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img, fname


def load_fold_models(summary_path, select_ckpt="f1"):
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Summary not found: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        fold_logs = json.load(f)

    models = []
    for item in fold_logs:
        if select_ckpt == "f1":
            order = ["ckpt_f1", "ckpt_used", "ckpt_loss"]
        elif select_ckpt == "loss":
            order = ["ckpt_loss", "ckpt_used", "ckpt_f1"]
        else:
            order = ["ckpt_used", "ckpt_f1", "ckpt_loss"]

        ckpt_path = next((p for k in order if (p := item.get(k)) and os.path.exists(p)), None)
        if not ckpt_path:
            raise FileNotFoundError(
                f"Checkpoint not found for fold {item.get('fold')}. "
                f"available={{'ckpt_used': {item.get('ckpt_used')}, 'ckpt_f1': {item.get('ckpt_f1')}, 'ckpt_loss': {item.get('ckpt_loss')}}}"
            )

        ckpt = torch.load(ckpt_path, map_location="cpu")
        arch = ckpt["arch"]; num_classes = ckpt["num_classes"]

        m = timm.create_model(arch, pretrained=False, num_classes=num_classes)
        m.load_state_dict(ckpt["model_state"], strict=True)
        m.eval()
        models.append(m)

        print(f"Loaded: {os.path.basename(ckpt_path)} | arch={arch} | num_classes={num_classes}")

    if not models:
        raise RuntimeError("No models loaded from summary.")
    return models


@torch.no_grad()
def infer_ensemble(models, loader, device, use_tta=False, return_proba=False, avg="logit"):
    
    assert avg in {"logit", "prob"}, "avg must be 'logit' or 'prob'"

    ids, preds, all_proba = [], [], []
    n_models = len(models)

    for xb, fnames in tqdm(loader, desc="Infer", leave=False):
        xb = xb.to(device, non_blocking=True)

        if avg == "logit":
            logit_sum = None
        else:
            prob_sum = None

        for m in models:
            m = m.to(device)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                if avg == "logit":
                    logit_sum_model, tta_count = None, 0
                    for v in _tta_stream(xb, enable=use_tta):
                        z = m(v) 
                        logit_sum_model = z if logit_sum_model is None else (logit_sum_model + z)
                        tta_count += 1
                    logit_avg_model = logit_sum_model / tta_count
                    logit_sum = logit_avg_model if logit_sum is None else (logit_sum + logit_avg_model)
                else:
                    prob_sum_model, tta_count = None, 0
                    for v in _tta_stream(xb, enable=use_tta):
                        p = torch.softmax(m(v), dim=1)
                        prob_sum_model = p if prob_sum_model is None else (prob_sum_model + p)
                        tta_count += 1
                    probs_model = prob_sum_model / tta_count
                    prob_sum = probs_model if prob_sum is None else (prob_sum + probs_model)

            m = m.to("cpu")
            torch.cuda.empty_cache()

        if avg == "logit":
            probs_avg = torch.softmax((logit_sum / n_models).float(), dim=1).cpu().numpy()
        else:
            probs_avg = (prob_sum / n_models).float().cpu().numpy()

        pred = probs_avg.argmax(axis=1)
        ids.extend(fnames)
        preds.extend(pred.tolist())
        if return_proba:
            all_proba.append(probs_avg)

    df = pd.DataFrame({"ID": ids, "target": preds})
    if return_proba:
        proba = np.concatenate(all_proba, axis=0)
        proba_cols = [f"prob_{i}" for i in range(proba.shape[1])]
        df = pd.concat([df, pd.DataFrame(proba, columns=proba_cols)], axis=1)
    return df


def _coerce_to_namespace(args):
    if args is None:
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
        ap.add_argument("--avg", type=str, default="logit", choices=["logit", "prob"])
        ap.add_argument("--select_ckpt", type=str, default="f1", choices=["f1", "loss"])
        return ap.parse_args()
    elif isinstance(args, dict):
        return SimpleNamespace(**args)
    else:
        return args


def run_inference(args=None):
    set_seed(42)
    args = _coerce_to_namespace(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    arch_dir = f"{args.arch}_{args.img_size}"
    summary_path = args.summary_path or os.path.join("model", arch_dir, "train_summary.json")

    models = load_fold_models(summary_path, select_ckpt=args.select_ckpt)

    model0 = models[0]
    data_cfg = resolve_model_data_config(model0)
    mean = tuple(map(float, data_cfg["mean"]))
    std = tuple(map(float, data_cfg["std"]))
    img_size = _resolve_img_size_from_model(model0, args.img_size)

    interp = INTER_AUTO
    ds = TestImageDataset(args.test_dir, transform=build_test_tf(img_size, mean, std, interpolation=interp))

    use_pin = torch.cuda.is_available()
    dl_kwargs = dict(
        dataset=ds,
        batch_size=args.infer_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_pin,
        persistent_workers=(args.num_workers > 0),
    )
    if args.num_workers > 0:
        dl_kwargs["prefetch_factor"] = 4
    dl = DataLoader(**dl_kwargs)

    df = infer_ensemble(models, dl, device, use_tta=args.tta, return_proba=args.save_proba, avg=getattr(args, "avg", "logit"))

    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, f"{arch_dir}.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    wandb_enabled = (wandb is not None) and (getattr(wandb, "run", None) is not None)
    if wandb_enabled:
        try:
            wandb.log({
                "inference/num_images": len(df),
                "inference/tta": bool(args.tta),
                "inference/save_proba": bool(args.save_proba),
                "inference/output_csv": out_csv,
            })
            cls_counts = df["target"].value_counts().to_dict()
            wandb.log({f"inference/class_count/{int(k)}": int(v) for k, v in cls_counts.items()})

            art = wandb.Artifact(f"{arch_dir}_pred", type="inference_csv")
            art.add_file(out_csv)
            wandb.log_artifact(art)
        except Exception:
            pass

          
if __name__ == "__main__":
    run_inference()
