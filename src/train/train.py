import os, json, random
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial

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
from src.evaluate.evaluate import macro_f1, count_mismatch_by_class


def _resolve_img_size(model, user_opt):
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


def build_model(arch = 'resnet50', num_classes = 17):
    model = timm.create_model(arch, pretrained=True, num_classes=num_classes)
    return model


def list_models(filter_name = "", pretrained = True):
    """
    timm에서 사용 가능한 모델 아키텍처(arch) 목록을 반환
    """
    pattern = f"*{filter_name}*" if filter_name and not any(ch in filter_name for ch in "*?[]") else filter_name
    names = timm.list_models(filter=pattern, pretrained=pretrained)
    return names


def set_seed(seed = 42):
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


class FocalLoss(nn.Module):
    def __init__(self, gamma = 1.5, weight = None, reduction = "mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# OOM 방지를 위해 accum_step 추가
def train_one_epoch(model, loader, criterion, optimizer, device,
                    ema: ModelEmaV2 | None = None, accum_steps = 1):
    model.train()
    epoch_loss, all_true, all_pred = 0.0, [], []
    optimizer.zero_grad(set_to_none=True)

    steps_done = 0
    for batch in tqdm(loader, desc="Train", leave=False):
        xb, yb = to_device(batch, device)

        with autocast(device_type='cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
            logits = model(xb)
            raw_loss = criterion(logits, yb)

        epoch_loss += raw_loss.item() * xb.size(0)
        loss = raw_loss / max(1, accum_steps)
        scaler.scale(loss).backward()
        steps_done += 1

        if (steps_done % accum_steps) == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)

        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_pred.extend(preds.tolist())
        all_true.extend(yb.detach().cpu().numpy().tolist())

    if (steps_done % accum_steps) != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if ema is not None:
            ema.update(model)

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
    def __init__(self, patience = 5, min_delta = 0.001):
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
    early_stopping_min_delta: float = 0.001,
    accum_steps: int = 1,
    use_focal: bool = False,
    focal_gamma: float = 1.5,
    save_oof_folds: bool = False
):
    set_seed(42)

    arch_dir = os.path.join(model_dir, f"{arch}_{img_size_opt}")    # save model under model/{model}_{img_size_opt}/
    os.makedirs(arch_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)  # save submission csv to output folder
    oof_dir = os.path.join(output_dir, "oof")
    os.makedirs(oof_dir, exist_ok=True)     # save all OOF csvs here

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_logs = []
    oof_chunks = []   # fold별 검증 확률 모음

    for fold, (tr_idx, va_idx) in enumerate(skf.split(df, df['target']), start=1):
        print(f"\n========== Fold {fold}/{n_folds} ==========")
        # 원본 인덱스 보존하여 OOF 병합 시 순서 복원
        tr_df = df.iloc[tr_idx].reset_index(drop=False).rename(columns={"index":"orig_idx"})
        va_df = df.iloc[va_idx].reset_index(drop=False).rename(columns={"index":"orig_idx"})

        model = build_model(arch=arch, num_classes=num_classes).to(device)
        data_cfg = resolve_model_data_config(model)
        mean = tuple(map(float, data_cfg['mean']))
        std  = tuple(map(float, data_cfg['std']))
        img_size = _resolve_img_size(model, img_size_opt)

        # image augmentation
        train_tf_base = build_train_tf_base(img_size, mean, std)
        train_tf_heavy = build_train_tf_heavy(img_size, mean, std)
        valid_tf_local = build_valid_tf(img_size, mean, std)

        ######### 이미지별 증강 기법 적용 추가 #########
        tf_doc = make_tf_doc(img_size, mean, std)
        tf_id  = make_tf_idcert(img_size, mean, std)
        tf_car = make_tf_car(img_size, mean, std)

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

        # 손실 함수 선택: 기본(LabelSmoothing), FocalLoss 전환 가능
        if use_focal:
            criterion = FocalLoss(gamma=focal_gamma)
        else:
            criterion = LabelSmoothingCrossEntropy(smoothing=0.01)

        optimizer = optim.AdamW(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        ema = ModelEmaV2(model, decay=0.999)

        best_f1, best_path = -1.0, os.path.join(arch_dir, f"fold{fold}_best.pt")
        history = []

        stopper = EarlyStopper(patience=early_stopping_patience, min_delta=early_stopping_min_delta) \
                  if use_early_stopping else None

        for epoch in range(1, epochs + 1):
            tr_loss, tr_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device, ema=ema, accum_steps=accum_steps)
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

        ############ fold마다 틀리게 예측한 class 개수 확인 ############
        ckpt = torch.load(best_path, map_location="cpu")
        ema.module.load_state_dict(ckpt["model_state"])

        all_true, all_pred = [], []
        ema.module.eval()
        with torch.no_grad():
            for xb, yb in tqdm(valid_loader, desc=f"Calculating Mismatch in Valid Fold{fold}", leave=False):
                xb, yb = to_device((xb, yb), device)
                with autocast(device_type='cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                    logits = ema.module(xb)
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                all_pred.extend(pred.tolist())
                all_true.extend(yb.cpu().numpy().tolist())

        mismatch_by_class = count_mismatch_by_class(all_true, all_pred)
        print(f"Valid Fold{fold} mismatch count by class: {mismatch_by_class}")

        if save_fold_logs:
            log_path = os.path.join(arch_dir, f"train_log_fold{fold}.csv")
            pd.DataFrame(history).to_csv(log_path, index=False)

        fold_logs.append({
            "fold": fold,
            "best_valid_f1": best_f1,
            "ckpt": best_path,
            "mismatch_by_class": {int(k): int(v) for k, v in mismatch_by_class.items()},
        })
        ############################################################


        ######################## OOF 확률 저장 ########################
        prob_list = []
        ema.module.eval()
        with torch.no_grad():
            for xb, yb in tqdm(valid_loader, desc=f"OOF Proba Fold{fold}", leave=False):
                xb, yb = to_device((xb, yb), device)
                with autocast(device_type='cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                    logits = ema.module(xb)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                prob_list.append(probs)
        prob_arr = np.concatenate(prob_list, axis=0)  # (N_valid, num_classes)
        prob_cols = [f"prob_{i}" for i in range(prob_arr.shape[1])]
        oof_fold_df = pd.DataFrame(prob_arr, columns=prob_cols)

        oof_fold_df.insert(0, "target", va_df["target"].values)     # valid_loader는 shuffle=False 이므로 va_df와 순서 동일 
        oof_fold_df.insert(0, "ID", va_df["ID"].astype(str).values)
        oof_fold_df["fold"] = fold
        oof_fold_df["orig_idx"] = va_df["orig_idx"].values

        # fold별 OOF 저장 옵션 켜지면 저장: output/oof/{arch}_{img}_fold{fold}.csv
        if save_oof_folds:
            arch_img = f"{arch}_{img_size_opt}"
            oof_fold_path = os.path.join(oof_dir, f"{arch_img}_fold{fold}.csv")
            oof_fold_df.to_csv(oof_fold_path, index=False)
            print(f"[OOF] Saved fold OOF: {oof_fold_path}")
        oof_chunks.append(oof_fold_df)

        del model, optimizer, scheduler, ema, train_loader, valid_loader, train_ds, valid_ds
        gc.collect()
        torch.cuda.empty_cache()
        ############################################################


    # save summary
    summary_path = os.path.join(arch_dir, "train_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(fold_logs, f, ensure_ascii=False, indent=2)

    # 모든 fold OOF 결합 및 저장 
    if oof_chunks:
        oof_df = pd.concat(oof_chunks, axis=0, ignore_index=True)
        # 컬럼을 원본 순서대로 정리
        oof_df = oof_df.sort_values("orig_idx").drop(columns=["orig_idx"])
        # output/oof/{arch}_{img_size_opt}_oof.csv 로 저장
        arch_img = f"{arch}_{img_size_opt}"
        oof_out = os.path.join(oof_dir, f"{arch_img}_oof.csv")
        oof_df.to_csv(oof_out, index=False)
        print(f"[OOF] Saved merged: {oof_out} (shape={oof_df.shape})")

    print("\nTraining finished.")
    print("Summary:", fold_logs)
    print(f"- Checkpoints saved under:  {arch_dir}")
    print(f"- Summary saved: {summary_path}")

    return summary_path
