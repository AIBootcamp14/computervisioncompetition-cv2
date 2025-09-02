import os, json, random
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

import gc
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from timm.data import resolve_model_data_config
from timm.loss import LabelSmoothingCrossEntropy
from timm.utils import ModelEmaV2
from sklearn.model_selection import StratifiedKFold

from src.data.transform import (
    build_train_tf_base, build_train_tf_heavy, build_valid_tf,
    make_tf_doc, make_tf_idcert, make_tf_car
)
from src.data.dataset import ImgDataset
from src.evaluate.evaluate import macro_f1


def _resolve_img_size(model, user_opt: str) -> int:
    data_cfg = resolve_model_data_config(model)
    _, in_h, in_w = data_cfg['input_size']

    # input: int
    if user_opt.isdigit():
        forced = int(user_opt)
        warnings.warn(f"Train stage: Override img_size to {forced} (model default: {in_h}x{in_w})", UserWarning)
        return forced

    # input: auto / auto-long
    if in_h != in_w:
        if user_opt == "auto-long":
            size = max(in_h, in_w)
            warnings.warn(f"Train stage: Rectangular default ({in_h}x{in_w}). Using longer side = {size}.", UserWarning)
            return size
        else:  # "auto"
            warnings.warn(f"Train stage: Rectangular default ({in_h}x{in_w}). Using height = {in_h}.", UserWarning)
            return in_h
    else:
        return in_h


def build_model(arch: str = 'resnet50', num_classes: int = 17):
    model = timm.create_model(arch, pretrained=True, num_classes=num_classes)
    return model


def list_models(filter_name: str = "", pretrained: bool = True):
    """
    timm에서 사용 가능한 모델 아키텍처(arch) 목록을 반환
    """
    pattern = f"*{filter_name}*" if filter_name and not any(ch in filter_name for ch in "*?[]") else filter_name
    names = timm.list_models(filter=pattern, pretrained=pretrained)
    return names


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed) 


def to_device(batch, device):
    xb, yb = batch
    return xb.to(device), yb.to(device)


scaler = GradScaler(enabled=torch.cuda.is_available())

def train_one_epoch(model, loader, criterion, optimizer, device, ema: ModelEmaV2 | None = None):
    model.train()
    epoch_loss, all_true, all_pred = 0.0, [], []
    for xb, yb in tqdm(loader, desc="Train", leave=False):
        xb, yb = to_device((xb, yb), device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type='cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
            logits = model(xb)
            loss = criterion(logits, yb)        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if ema is not None:
            ema.update(model)

        epoch_loss += loss.item() * xb.size(0)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_pred.extend(preds.tolist())
        all_true.extend(yb.detach().cpu().numpy().tolist())

    avg_loss = epoch_loss / len(loader.dataset)
    f1 = macro_f1(all_true, all_pred)
    return avg_loss, f1

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    epoch_loss, all_true, all_pred = 0.0, [], []
    for xb, yb in tqdm(loader, desc="Valid", leave=False):
        xb, yb = to_device((xb, yb), device)
        with autocast(device_type='cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
            logits = model(xb)
            loss = criterion(logits, yb)
        epoch_loss += loss.item() * xb.size(0)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_pred.extend(preds.tolist())
        all_true.extend(yb.cpu().numpy().tolist())

    avg_loss = epoch_loss / len(loader.dataset)
    f1 = macro_f1(all_true, all_pred)
    return avg_loss, f1


class EarlyStopper:
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        """
        patience 동안 valid_f1 이 min_delta 이상 개선되지 않으면 stop.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.num_bad_epochs = 0

    def step(self, current: float) -> bool:
        """
        returns: True if should stop now.
        """
        if self.best is None or (current - self.best) > self.min_delta:
            self.best = current
            self.num_bad_epochs = 0
            return False
        else:
            self.num_bad_epochs += 1
            return self.num_bad_epochs >= self.patience


def run_train(
    df: pd.DataFrame,
    device: torch.device,
    model_dir: str,
    output_dir: str,
    arch: str = 'resnet50',
    num_classes: int = 17,
    img_size_opt: str = "auto",
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 3e-4,
    n_folds: int = 5,
    minor_classes = (1, 13, 14),
    p_heavy_minor: float = 0.7,
    num_workers: int = 4,
    save_fold_logs: bool = False,   # save fold log as csv
    use_early_stopping: bool = True,
    early_stopping_patience: int = 5,
    early_stopping_min_delta: float = 0.001
):
    set_seed(42)

    arch_dir = os.path.join(model_dir, arch)    # save model under model/{model}/
    os.makedirs(arch_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)  # save submission csv to output folder

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_logs = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(df, df['target']), start=1):
        print(f"\n========== Fold {fold}/{n_folds} ==========")
        tr_df = df.iloc[tr_idx].reset_index(drop=True)
        va_df = df.iloc[va_idx].reset_index(drop=True)

        model = build_model(arch=arch, num_classes=num_classes).to(device)
        img_size = _resolve_img_size(model, img_size_opt)

        # image augmentation
        train_tf_base = build_train_tf_base(img_size)
        train_tf_heavy = build_train_tf_heavy(img_size)
        valid_tf_local = build_valid_tf(img_size)

        ######### 이미지별 증강 기법 적용 추가 #########
        tf_doc = make_tf_doc(img_size)
        tf_id  = make_tf_idcert(img_size)
        tf_car = make_tf_car(img_size)

        DOC_CLASSES = {0,1,3,4,6,7,10,11,12,13,14}
        ID_CERT_CLASSES = {5,8,9,15}
        CAR_CLASSES = {2,16}

        class_tf_map = {}
        for c in DOC_CLASSES: class_tf_map[c] = tf_doc
        for c in ID_CERT_CLASSES: class_tf_map[c] = tf_id
        for c in CAR_CLASSES: class_tf_map[c] = tf_car
        #############################################

        train_ds = ImgDataset(
            tr_df,
            tf_base=train_tf_base,
            tf_heavy=train_tf_heavy,
            minor_classes=set(minor_classes),
            p_heavy_minor=p_heavy_minor,
            use_heavy_for_all=False,
            base_dir="data/train",
            class_tf_map=class_tf_map,
        )

        valid_ds = ImgDataset(va_df, tf_base=valid_tf_local, tf_heavy=None, base_dir="data/train")

        class_counts = np.bincount(tr_df['target'].values, minlength=num_classes)
        w = (class_counts.mean() / (class_counts + 1e-6))
        w = np.sqrt(w)
        sample_weights = tr_df['target'].map(lambda y: w[y]).values
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=False,  # shuffle=False when using sampler
            sampler=sampler, num_workers=num_workers, pin_memory=True
        )
        valid_loader = DataLoader(
            valid_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

        criterion = LabelSmoothingCrossEntropy(smoothing=0.01)
        # criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        ema = ModelEmaV2(model, decay=0.999)

        best_f1, best_path = -1.0, os.path.join(arch_dir, f"fold{fold}_best.pt")
        history = []

        stopper = EarlyStopper(patience=early_stopping_patience, min_delta=early_stopping_min_delta) \
                  if use_early_stopping else None

        for epoch in range(1, epochs + 1):
            tr_loss, tr_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device, ema=ema)
            va_loss, va_f1 = validate(ema.module, valid_loader, criterion, device)
            scheduler.step()

            print(f"[Fold {fold}] Epoch {epoch:02d} | "
                  f"train_loss {tr_loss:.4f} f1 {tr_f1:.4f} || "
                  f"valid_loss {va_loss:.4f} f1 {va_f1:.4f}")

            history.append({
                "fold": fold, "epoch": epoch,
                "train_loss": tr_loss, "train_f1": tr_f1,
                "valid_loss": va_loss, "valid_f1": va_f1,
                "lr": scheduler.get_last_lr()[0]
            })

            # save best checkpoint for each fold
            if va_f1 > best_f1:
                best_f1 = va_f1
                torch.save({
                    "model_state": ema.module.state_dict(),
                    "arch": arch,
                    "num_classes": num_classes,
                    "fold": fold,
                    "epoch": epoch,
                    "best_valid_f1": best_f1,
                }, best_path)

            if stopper is not None and stopper.step(va_f1):
                print(f"[Fold {fold}] Early stopping triggered at epoch {epoch}. "
                      f"(best_valid_f1={best_f1:.4f})")
                break

        if save_fold_logs:
            log_path = os.path.join(arch_dir, f"train_log_fold{fold}.csv")
            pd.DataFrame(history).to_csv(log_path, index=False)

        fold_logs.append({
            "fold": fold,
            "best_valid_f1": best_f1,
            "ckpt": best_path,
        })

        del model, optimizer, scheduler, ema, train_loader, valid_loader, train_ds, valid_ds
        gc.collect()
        torch.cuda.empty_cache()

    # save summary
    summary_path = os.path.join(arch_dir, "train_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(fold_logs, f, ensure_ascii=False, indent=2)

    print("\nTraining finished.")
    print("Summary:", fold_logs)
    print(f"- Checkpoints saved under:  {arch_dir}")
    print(f"- Summary saved: {summary_path}")

    return summary_path
