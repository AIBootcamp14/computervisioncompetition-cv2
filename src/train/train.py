import os, json, random
import math
import warnings
import numpy as np
import pandas as pd
import tempfile
from dataclasses import dataclass
from tqdm import tqdm

import gc
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from timm.data import resolve_model_data_config
from timm.loss import LabelSmoothingCrossEntropy
from timm.utils import ModelEmaV2
from sklearn.model_selection import StratifiedKFold

from src.data.transform import (
    INTER_AUTO, build_train_tf_base, build_train_tf_heavy, ProbTransform, build_test_tf, build_valid_tf_robust,
    make_tf_doc, make_tf_doc_heavy, OverlayFromPool, make_tf_idcert, make_tf_car
)
from src.data.dataset import ImgDataset, CachedDataset
from src.train.schedulers import CosineAnnealingWarmupRestarts
from src.evaluate.evaluate import macro_f1, count_mismatch_by_class

try:
    import wandb
except Exception:
    wandb = None


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    try:
        import albumentations as A
        A.set_seed(worker_seed)
    except Exception:
        pass


def _resolve_img_size(model, user_opt):
    data_cfg = resolve_model_data_config(model)
    _, in_h, in_w = data_cfg['input_size']

    if user_opt.isdigit():
        forced = int(user_opt)
        warnings.warn(f"Train stage: Override img_size to {forced} (model default: {in_h}x{in_w})", UserWarning)
        return forced

    if in_h != in_w:
        if user_opt == "auto-long":
            size = max(in_h, in_w)
            warnings.warn(f"Train stage: Rectangular default ({in_h}x{in_w}). Using longer side = {size}.", UserWarning)
            return size
        else:
            warnings.warn(f"Train stage: Rectangular default ({in_h}x{in_w}). Using height = {in_h}.", UserWarning)
            return in_h
    else:
        return in_h


def build_model(arch='resnet50', num_classes=17):
    model = timm.create_model(arch, pretrained=True, num_classes=num_classes)
    return model


def list_models(filter_name="", pretrained=True):
    pattern = f"*{filter_name}*" if filter_name and not any(ch in filter_name for ch in "*?[]") else filter_name
    names = timm.list_models(filter=pattern, pretrained=pretrained)
    return names


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def to_device(batch, device):
    xb, yb = batch
    xb = xb.to(device, dtype=torch.float32, non_blocking=True)
    xb = xb.contiguous(memory_format=torch.channels_last)
    return xb, yb.to(device, non_blocking=True)


class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, weight=None, reduction="mean"):
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


def _serialize_focal_weight(fw):
    if fw is None:
        return None
    if isinstance(fw, str):
        return fw 
    if isinstance(fw, (list, np.ndarray)):
        return np.asarray(fw, dtype=np.float32).tolist()
    return str(fw)


def train_one_epoch(model, loader, criterion, optimizer, device, logit_adj, scaler, ema: ModelEmaV2 | None = None,
                    accum_steps=1, scheduler=None, clip_value=2.0):
    model.train()
    epoch_loss, all_true, all_pred = 0.0, [], []
    optimizer.zero_grad(set_to_none=True)

    steps_done = 0
    for batch in tqdm(loader, desc="Train", leave=False):
        xb, yb = to_device(batch, device)

        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=torch.cuda.is_available()):
            logits = model(xb)
            logits4loss = logits - logit_adj.view(1, -1).to(logits.dtype) if logit_adj is not None else logits
            raw_loss = criterion(logits4loss, yb)

        epoch_loss += raw_loss.item() * xb.size(0)
        loss = raw_loss / max(1, accum_steps)
        scaler.scale(loss).backward()
        steps_done += 1

        if (steps_done % accum_steps) == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)

        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_pred.extend(preds.tolist())
        all_true.extend(yb.detach().cpu().numpy().tolist())

    if (steps_done % accum_steps) != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if ema is not None:
            ema.update(model)

    avg_loss = epoch_loss / len(loader.dataset)
    f1 = macro_f1(all_true, all_pred)
    return avg_loss, f1


@torch.no_grad()
def validate(model, loader, criterion, device, logit_adj=None):
    model.eval()
    epoch_loss, all_true, all_pred = 0.0, [], []
    for xb, yb in tqdm(loader, desc="Valid", leave=False):
        xb, yb = to_device((xb, yb), device)
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=torch.cuda.is_available()):
            logits = model(xb)
            logits4loss = logits - logit_adj.view(1, -1).to(logits.dtype) if logit_adj is not None else logits
            loss = criterion(logits4loss, yb)
        epoch_loss += loss.item() * xb.size(0)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_pred.extend(preds.tolist())
        all_true.extend(yb.cpu().numpy().tolist())

    avg_loss = epoch_loss / len(loader.dataset)
    f1 = macro_f1(all_true, all_pred)
    return avg_loss, f1


@dataclass
class StopperResult:
    is_new_best: bool
    improved_loss: bool
    improved_f1: bool
    should_stop: bool
    best_loss: float
    best_f1_overall: float
    best_f1_at_best_loss: float
    best_epoch: int


class EarlyStopper:
    def __init__(self, patience=10, min_delta_loss=0.001, min_delta_f1=0.0003, mode="either"):
        self.patience = patience
        self.min_delta_loss = min_delta_loss
        self.min_delta_f1 = min_delta_f1
        self.mode = mode
        self.best_loss = None
        self.best_f1_overall = -1.0
        self.best_f1_at_best_loss = -1.0
        self.best_epoch = -1
        self.num_bad_epochs = 0

    def step(self, epoch, current_loss, current_f1, in_warmup=False):
        if self.best_loss is None:
            self.best_loss = current_loss
            self.best_f1_overall = current_f1
            self.best_f1_at_best_loss = current_f1
            self.best_epoch = epoch
            self.num_bad_epochs = 0
            return StopperResult(
                is_new_best=True, improved_loss=True, improved_f1=True,
                should_stop=False,
                best_loss=self.best_loss,
                best_f1_overall=self.best_f1_overall,
                best_f1_at_best_loss=self.best_f1_at_best_loss,
                best_epoch=self.best_epoch,
            )

        loss_improved = (self.best_loss - current_loss) > self.min_delta_loss
        f1_improved = (current_f1 - self.best_f1_overall) > self.min_delta_f1

        if self.mode == "loss":
            improved = loss_improved
        elif self.mode == "f1":
            improved = f1_improved
        elif self.mode == "both":
            improved = loss_improved and f1_improved
        elif self.mode == "either":
            improved = loss_improved or f1_improved
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        is_new_best = False
        if improved:
            is_new_best = True
            if loss_improved:
                self.best_loss = current_loss
                self.best_f1_at_best_loss = current_f1
            if f1_improved:
                self.best_f1_overall = current_f1
            self.best_epoch = epoch
            self.num_bad_epochs = 0
        else:
            if in_warmup:
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

        should_stop = (self.num_bad_epochs >= self.patience)

        return StopperResult(
            is_new_best=is_new_best,
            improved_loss=loss_improved,
            improved_f1=f1_improved,
            should_stop=should_stop,
            best_loss=self.best_loss,
            best_f1_overall=self.best_f1_overall,
            best_f1_at_best_loss=self.best_f1_at_best_loss,
            best_epoch=self.best_epoch,
        )


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
    accum_steps: int = 1,
    lr: float = 3e-4,
    n_folds: int = 5,
    minor_classes=(1, 13, 14),
    p_heavy_minor: float = 0.7,
    doc_heavy_probs: dict = None,
    num_workers: int = 4,
    save_fold_logs: bool = False,
    use_early_stopping: bool = True,
    early_stopping_patience: int = 5,
    early_stopping_mode: str = "either",
    early_stopping_min_delta_loss: float = 0.001,
    early_stopping_min_delta_f1: float = 0.0005,
    eval_mode: str = "f1",
    base_clip_value: float = 2.0,
    use_focal: bool = False,
    focal_gamma: float = 2,
    focal_weight: str | np.ndarray | list | None = None,
    use_logit_adjustment: bool = False,
    logit_tau: float = 1.0,
    save_oof_folds: bool = False,
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
):

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    set_seed(42)


    wandb_enabled = bool(wandb_project and (wandb is not None))
    if not wandb_enabled:
        warnings.warn(f"[wandb] disabled. project={wandb_project} entity={wandb_entity} module={'ok' if wandb else 'missing'}", UserWarning)

    base_wandb_name = (wandb_run_name or os.environ.get("WANDB_NAME") or f"{arch}_{img_size_opt}")
    wandb_group = os.environ.get("WANDB_GROUP") or base_wandb_name
    wandb_tags = (os.environ.get("WANDB_TAGS") or "").split(",") if os.environ.get("WANDB_TAGS") else None

    arch_dir = os.path.join(model_dir, f"{arch}_{img_size_opt}")
    os.makedirs(arch_dir, exist_ok=True)    
    os.makedirs(output_dir, exist_ok=True)
    oof_dir = os.path.join(output_dir, "oof")
    os.makedirs(oof_dir, exist_ok=True)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_logs = []
    oof_chunks = []


    for fold, (tr_idx, va_idx) in enumerate(skf.split(df, df['target']), start=1):

        g = torch.Generator()
        g.manual_seed(42)

        scaler = GradScaler(enabled=torch.cuda.is_available())

        best_loss_path = os.path.join(arch_dir, f"fold{fold}_best_loss.pt")
        best_f1_path   = os.path.join(arch_dir, f"fold{fold}_best_f1.pt")

        print(f"\n========== Fold {fold}/{n_folds} ==========")
        tr_df = df.iloc[tr_idx].reset_index(drop=False).rename(columns={"index": "orig_idx"})
        va_df = df.iloc[va_idx].reset_index(drop=False).rename(columns={"index": "orig_idx"})

        tr_ids = set(tr_df["ID"].astype(str).tolist())
        va_ids = set(va_df["ID"].astype(str).tolist())
        _overlap = tr_ids & va_ids
        assert len(_overlap) == 0, f"[LEAK] Train/Valid share {len(_overlap)} IDs, e.g. {list(sorted(_overlap))[:5]}"

        model = build_model(arch=arch, num_classes=num_classes).to(device)
        model = model.to(memory_format=torch.channels_last)

        clip_value = 5.0 if any(x in arch.lower() for x in ["huge", "large", "xlarge"]) else base_clip_value

        data_cfg = resolve_model_data_config(model)
        mean = tuple(map(float, data_cfg['mean']))
        std = tuple(map(float, data_cfg['std']))
        img_size = _resolve_img_size(model, img_size_opt)

        interp = INTER_AUTO

        train_tf_base = build_train_tf_base(img_size, mean, std, interpolation=interp)
        train_tf_heavy = build_train_tf_heavy(img_size, mean, std, interpolation=interp)
        valid_tf_local = build_test_tf(img_size, mean, std, interpolation=interp)
        valid_tf_robust = build_valid_tf_robust(img_size, mean, std, interpolation=interp)

        DOC_CLASSES = {0, 1, 3, 4, 6, 7, 10, 11, 12, 13, 14}
        ID_CERT_CLASSES = {5, 8, 9, 15}
        CAR_CLASSES = {2, 16}

        doc_train_paths = [os.path.join("data/train", str(row.ID)) for _, row in tr_df.iterrows() if row.target in DOC_CLASSES]

        doc_overlay_base = OverlayFromPool(
            paths=doc_train_paths,
            alpha=(0.06, 0.14),
            angle_set=(0, 90, 180, 270),
            scale=(0.9, 1.1),
            self_or_pool_prob=0.6,
            p=0.1
        )
        doc_overlay_heavy = OverlayFromPool(
            paths=doc_train_paths,
            alpha=(0.06, 0.14),
            angle_set=(0, 90, 180, 270),
            scale=(0.9, 1.1),
            self_or_pool_prob=0.6,
            p=0.04 
        )
        
        tf_doc = make_tf_doc(img_size, mean, std, interpolation=interp, overlay=doc_overlay_base)
        tf_doc_heavy = make_tf_doc_heavy(img_size, mean, std, interpolation=interp, overlay=doc_overlay_heavy)
        tf_id = make_tf_idcert(img_size, mean, std, interpolation=interp)
        tf_car = make_tf_car(img_size, mean, std, interpolation=interp)

        p_doc_heavy_base = {}
        default_p_doc_heavy = 0.4

        if doc_heavy_probs:
            p_doc_heavy_base.update(doc_heavy_probs)

        p_doc_heavy = p_doc_heavy_base

        class_tf_map = {}
        for c in DOC_CLASSES:
            class_tf_map[c] = ProbTransform(tf_doc, tf_doc_heavy, p=p_doc_heavy.get(c, default_p_doc_heavy))
        for c in ID_CERT_CLASSES:
            class_tf_map[c] = tf_id
        for c in CAR_CLASSES:
            class_tf_map[c] = tf_car

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

        valid_ds_robust = ImgDataset(va_df, tf_base=valid_tf_robust, tf_heavy=None, base_dir="data/train")
        valid_ds_robust = CachedDataset(valid_ds_robust)

        class_counts = np.bincount(tr_df['target'].values, minlength=num_classes)
        w = (class_counts.mean() / (class_counts + 1e-6))
        w = np.sqrt(w)

        prior = class_counts / class_counts.sum()
        log_prior = torch.log(torch.tensor(prior, dtype=torch.float32, device=device).clamp_min(1e-12))
        logit_adj = logit_tau * log_prior if use_logit_adjustment else None

        alpha_cb_t = None
        if isinstance(focal_weight, str) and focal_weight.lower() in {"alpha_cb", "cb"}:
            beta = 0.99
            eff_num = 1.0 - np.power(beta, class_counts.clip(min=1))
            alpha_cb = (1.0 - beta) / np.maximum(eff_num, 1e-12)
            alpha_cb = alpha_cb * (len(alpha_cb) / alpha_cb.sum())
            alpha_cb_t = torch.tensor(alpha_cb, dtype=torch.float32, device=device)
        elif isinstance(focal_weight, (list, np.ndarray)):
            alpha_arr = np.asarray(focal_weight, dtype=np.float32)
            assert alpha_arr.shape[0] == num_classes, "focal_weight length must equal num_classes"
            alpha_cb_t = torch.tensor(alpha_arr, dtype=torch.float32, device=device)

        y = tr_df['target'].values
        sample_weights = torch.as_tensor(w[y], dtype=torch.double)
        sampler = WeightedRandomSampler(weights=sample_weights,
                                        num_samples=len(sample_weights),
                                        replacement=True,
                                        generator=g)

        use_pin = torch.cuda.is_available()

        dl_kwargs = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=use_pin,
            persistent_workers=(num_workers > 0),
            worker_init_fn=seed_worker,
            generator=g,
        )
        if num_workers > 0:
            dl_kwargs["prefetch_factor"] = 4

        train_loader = DataLoader(train_ds, shuffle=False, sampler=sampler, **dl_kwargs)

        valid_dl_kwargs = dict(
            batch_size=batch_size,
            num_workers=max(0, min(2, num_workers)),
            pin_memory=use_pin,
            persistent_workers=False,
            worker_init_fn=seed_worker,
        )
        if valid_dl_kwargs["num_workers"] > 0:
            valid_dl_kwargs["prefetch_factor"] = 4
        valid_loader = DataLoader(valid_ds, shuffle=False, **valid_dl_kwargs)
        valid_loader_robust = DataLoader(valid_ds_robust, shuffle=False, **valid_dl_kwargs)

        wrun = None
        if wandb_enabled:
            wrun = wandb.init(
                entity=wandb_entity or os.environ.get("WANDB_ENTITY"),
                project=wandb_project or os.environ.get("WANDB_PROJECT"),
                name=f"{base_wandb_name}-fold{fold}",
                group=wandb_group,
                tags=wandb_tags,
                resume="allow",
                config={
                    "arch": arch,
                    "img_size": img_size_opt,
                    "fold": fold,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "accum_steps": accum_steps,
                    "use_focal": use_focal,
                    "focal_gamma": focal_gamma,
                    "focal_weight": _serialize_focal_weight(focal_weight), 
                    "use_logit_adjustment": use_logit_adjustment,
                    "logit_tau": logit_tau, 
                    "early_stopping": use_early_stopping,
                    "early_stopping_patience": early_stopping_patience,
                    "early_stopping_mode": early_stopping_mode,
                    "early_stopping_min_delta_loss": early_stopping_min_delta_loss,
                    "early_stopping_min_delta_f1": early_stopping_min_delta_f1,
                    "n_folds": n_folds,
                }
            )
            wandb.define_metric("epoch")
            wandb.define_metric("lr", step_metric="epoch")
            wandb.define_metric("*", step_metric="epoch")

        if use_focal:
            weight_t = alpha_cb_t
            criterion = FocalLoss(gamma=focal_gamma, weight=weight_t)
        else:
            criterion = LabelSmoothingCrossEntropy(smoothing=0.05)

        focal_weight_used = ("none" if not use_focal else
            ("alpha_cb" if (isinstance(focal_weight, str) and focal_weight.lower() in {"alpha_cb", "cb"}) else
            ("custom" if isinstance(focal_weight, (list, np.ndarray)) else "none"))
        )
        if wandb_enabled and wrun is not None:
            wrun.config.update({"focal_weight_used": focal_weight_used}, allow_val_change=True)

        try:
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02, foreach=True)
        except TypeError:
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)

        num_train = len(tr_df)

        steps_per_epoch = math.ceil(num_train / batch_size)
        opt_steps_per_epoch = math.ceil(steps_per_epoch / max(1, accum_steps))
        total_opt_steps = opt_steps_per_epoch * epochs

        warmup_ratio = 0.1 if any(x in arch.lower() for x in ["huge", "large", "xlarge"]) else 0.05
        warmup_steps = max(200, min(800, int(total_opt_steps * warmup_ratio)))
        warmup_steps = max(0, min(warmup_steps, max(1, total_opt_steps - 1)))
        min_lr = max(lr * 1e-2, 1e-6)

        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=total_opt_steps,
            cycle_mult=1.0,
            max_lr=lr,
            min_lr=min_lr,
            warmup_steps=warmup_steps,
            gamma=1.0
        )
        warmup_epochs = min(epochs - 1, math.ceil(warmup_steps / max(1, opt_steps_per_epoch)))

        if wandb_enabled and wrun is not None:
            wrun.config.update({
                "opt_steps_per_epoch": int(opt_steps_per_epoch),
                "total_opt_steps": int(total_opt_steps),
                "warmup_steps": int(warmup_steps),
                "warmup_epochs": int(warmup_epochs),
                "eval_mode": eval_mode,
            }, allow_val_change=True)

        ema = ModelEmaV2(model, decay=0.999)
        history = []

        tracker = EarlyStopper(
            patience=early_stopping_patience,
            min_delta_loss=early_stopping_min_delta_loss,
            min_delta_f1=early_stopping_min_delta_f1,
            mode=early_stopping_mode
        )

        for epoch in range(1, epochs + 1):
            tr_loss, tr_f1 = train_one_epoch(
                model, train_loader, criterion, optimizer, device, logit_adj=logit_adj, scaler=scaler,
                ema=ema, accum_steps=accum_steps, scheduler=scheduler, clip_value=clip_value,
            )
            va_loss, va_f1 = validate(ema.module, valid_loader, criterion, device, logit_adj=logit_adj)
            robust_va_loss, robust_va_f1 = validate(ema.module, valid_loader_robust, criterion, device, logit_adj=logit_adj)

            in_warmup = (epoch <= warmup_epochs)
            if in_warmup:
                print(f"[Fold {fold}] (WARMUP {epoch}/{warmup_epochs})", end=" ")

            res = tracker.step(epoch, va_loss, va_f1, in_warmup=in_warmup)

            if eval_mode == "loss":
                if res.improved_loss:
                    save_meta = {
                        "model_state": ema.module.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "arch": arch,
                        "num_classes": num_classes,
                        "fold": fold,
                        "epoch": epoch,
                        "save_best_by": "loss",
                        "best_valid_loss": float(res.best_loss),
                        "best_valid_f1_overall": float(res.best_f1_overall),
                        "best_valid_f1_at_best_loss": float(res.best_f1_at_best_loss),
                        "hparams": {
                            "img_size_opt": img_size_opt,
                            "resolved_img_size": int(img_size),
                            "mean": list(mean),
                            "std": list(std),
                            "lr": lr,
                            "epochs": epochs,
                            "accum_steps": accum_steps,
                            "use_logit_adjustment": bool(use_logit_adjustment),
                            "logit_tau": float(logit_tau),
                            "use_focal": bool(use_focal),
                            "focal_gamma": float(focal_gamma),
                            "focal_weight": _serialize_focal_weight(focal_weight),
                            "focal_weight_used": focal_weight_used,
                            "class_counts": class_counts.tolist(),
                            "seed": 42,
                        },
                    }
                    torch.save(save_meta, best_loss_path)
            elif eval_mode == "f1":
                if res.improved_f1:
                    save_meta_f1 = {
                        "model_state": ema.module.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "arch": arch,
                        "num_classes": num_classes,
                        "fold": fold,
                        "epoch": epoch,
                        "save_best_by": "f1_overall",
                        "best_valid_loss": float(va_loss),
                        "best_valid_f1_overall": float(res.best_f1_overall),
                        "best_valid_f1_at_best_loss": float(res.best_f1_at_best_loss),
                        "hparams": {
                            "img_size_opt": img_size_opt,
                            "resolved_img_size": int(img_size),
                            "mean": list(mean),
                            "std": list(std),
                            "lr": lr,
                            "epochs": epochs,
                            "accum_steps": accum_steps,
                            "use_logit_adjustment": bool(use_logit_adjustment),
                            "logit_tau": float(logit_tau),
                            "use_focal": bool(use_focal),
                            "focal_gamma": float(focal_gamma),
                            "focal_weight": _serialize_focal_weight(focal_weight),
                            "focal_weight_used": focal_weight_used,
                            "class_counts": class_counts.tolist(),
                            "seed": 42,
                        },
                    }
                    torch.save(save_meta_f1, best_f1_path)

            print(f"[Fold {fold}] Epoch {epoch:02d} | "
                  f"train_loss {tr_loss:.4f} f1 {tr_f1:.4f} || "
                  f"valid_loss {va_loss:.4f} f1 {va_f1:.4f} || "
                  f"robust_valid_loss {robust_va_loss:.4f} f1 {robust_va_f1:.4f} || "
                  f"is_new_best={res.is_new_best} (best_epoch={tracker.best_epoch})")
         
            if wandb_enabled:
                wandb.log({
                    "epoch": epoch,
                    "train/loss": tr_loss,
                    "train/f1": tr_f1,
                    "clean_valid/loss": va_loss,
                    "clean_valid/f1": va_f1,
                    "robust_valid/loss": robust_va_loss,
                    "robust_valid/f1": robust_va_f1,
                    "lr": optimizer.param_groups[0]['lr'],
                }, step=epoch)

            history.append({
                "fold": fold, "epoch": epoch,
                "train_loss": tr_loss, "train_f1": tr_f1,
                "valid_loss": va_loss, "valid_f1": va_f1,
                "lr": optimizer.param_groups[0]['lr']
            })

            if use_early_stopping and res.should_stop:
                print(f"[Fold {fold}] Early stopping at epoch {epoch}")
                break

        load_path = best_f1_path if (eval_mode == "f1" and os.path.exists(best_f1_path)) else best_loss_path
        ckpt = torch.load(load_path, map_location="cpu")
        ema.module.load_state_dict(ckpt["model_state"])
        eval_model = ema.module

        all_true, all_pred = [], []
        eval_model.eval()
        with torch.no_grad():
            for xb, yb in tqdm(valid_loader, desc=f"Calculating Mismatch in Valid Fold{fold}", leave=False):
                xb, yb = to_device((xb, yb), device)
                with torch.cuda.amp.autocast(dtype=torch.float16, enabled=torch.cuda.is_available()):
                    logits = eval_model(xb)
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                all_pred.extend(pred.tolist())
                all_true.extend(yb.cpu().numpy().tolist())

        mismatch_by_class = count_mismatch_by_class(all_true, all_pred)
        print(f"Valid Fold{fold} mismatch count by class: {mismatch_by_class}")

        if save_fold_logs:
            log_path = os.path.join(arch_dir, f"train_log_fold{fold}.csv")
            pd.DataFrame(history).to_csv(log_path, index=False)

        if wandb_enabled:
            try:          
                rows = [(int(k), int(v)) for k, v in sorted(mismatch_by_class.items())]
                mm_table = wandb.Table(columns=["class", "mismatch_count"], data=rows)
                wandb.log({"valid_mismatch/table_by_class": mm_table})

                wandb.run.summary.update({f"valid_mismatch/by_class/{k}": int(v) for k, v in mismatch_by_class.items()})
                
                cm_plot = wandb.plot.confusion_matrix(
                    y_true=all_true,
                    preds=all_pred,
                    class_names=[str(i) for i in range(num_classes)]
                )
                wandb.log({f"valid_mismatch/confusion_matrix": cm_plot})
            except Exception:
                pass


        ckpt_loss = best_loss_path if os.path.exists(best_loss_path) else None
        ckpt_f1   = best_f1_path   if os.path.exists(best_f1_path)   else None
        ckpt_used = ckpt_f1 if (eval_mode == "f1" and ckpt_f1) else ckpt_loss

        fold_logs.append({
            "fold": fold,
            "best_valid_loss": float(tracker.best_loss),
            "best_valid_f1_overall": float(tracker.best_f1_overall),
            "best_valid_f1_at_best_loss": float(tracker.best_f1_at_best_loss),

            "best_epoch_es": int(tracker.best_epoch),
            "early_stopping_mode": early_stopping_mode,

            "eval_mode_used": eval_mode,
            "ckpt_used": ckpt_used,
            "ckpt_loss": ckpt_loss,
            "ckpt_f1": ckpt_f1,

            "mismatch_by_class": {int(k): int(v) for k, v in mismatch_by_class.items()},
        })

        if wandb_enabled:
            wandb.run.summary["best_valid_loss"] = float(tracker.best_loss)
            wandb.run.summary["best_valid_f1_overall"] = float(tracker.best_f1_overall)
            wandb.run.summary["best_valid_f1_at_best_loss"] = float(tracker.best_f1_at_best_loss)
            wandb.run.summary["best_epoch"] = int(tracker.best_epoch)
            wandb.run.summary["saved_by"] = eval_mode

        if wandb_enabled:
            try:
                wandb_config = {
                    "model_state": ema.module.state_dict(),
                    "arch": arch,
                    "num_classes": num_classes,
                    "fold": fold,
                    "epoch": int(tracker.best_epoch),
                    "saved_by": eval_mode,
                    "best_valid_loss": float(tracker.best_loss),
                    "best_valid_f1_overall": float(tracker.best_f1_overall),
                    "best_valid_f1_at_best_loss": float(tracker.best_f1_at_best_loss),
                    "hparams": {
                        "img_size_opt": img_size_opt,
                        "resolved_img_size": int(img_size),
                        "mean": list(mean),
                        "std": list(std),
                        "use_logit_adjustment": bool(use_logit_adjustment),
                        "logit_tau": float(logit_tau),
                        "use_focal": bool(use_focal),
                        "focal_gamma": float(focal_gamma),
                        "focal_weight": _serialize_focal_weight(focal_weight),
                        "focal_weight_used": focal_weight_used,
                        "class_counts": class_counts.tolist(),
                        "seed": 42
                    }
                }

                with tempfile.TemporaryDirectory() as tmpd:
                    wandb_upload_path = os.path.join(tmpd, f"fold{fold}_best_{eval_mode}_light.pt")
                    torch.save(wandb_config, wandb_upload_path)
                    art = wandb.Artifact(
                        f"{arch}_{img_size_opt}_fold{fold}_light",
                        type="model",
                        metadata={
                            "best_epoch": int(tracker.best_epoch),
                            "best_valid_loss": float(tracker.best_loss),
                            "best_valid_f1_at_best_loss": float(tracker.best_f1_at_best_loss),
                            "best_valid_f1_overall": float(tracker.best_f1_overall),
                            "saved_by": eval_mode,
                            "arch": arch,
                            "img_size_opt": img_size_opt,
                            "resolved_img_size": int(img_size),
                        }
                    )
                    art.add_file(wandb_upload_path)

                    logged = wandb.log_artifact(art, aliases=["best", f"fold{fold}", f"{eval_mode}_light"])

                    try:
                        logged.wait()
                    except Exception:
                        pass

            except Exception:
                pass

        prob_list = []
        ema.module.eval()
        with torch.no_grad():
            for xb, yb in tqdm(valid_loader, desc=f"OOF Proba Fold{fold}", leave=False):
                xb, yb = to_device((xb, yb), device)
                with torch.cuda.amp.autocast(dtype=torch.float16, enabled=torch.cuda.is_available()):
                    logits = ema.module(xb)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                prob_list.append(probs)
        prob_arr = np.concatenate(prob_list, axis=0) 
        prob_cols = [f"prob_{i}" for i in range(prob_arr.shape[1])]
        oof_fold_df = pd.DataFrame(prob_arr, columns=prob_cols)

        oof_fold_df.insert(0, "target", va_df["target"].values) 
        oof_fold_df.insert(0, "ID", va_df["ID"].astype(str).values)
        oof_fold_df["fold"] = fold
        oof_fold_df["orig_idx"] = va_df["orig_idx"].values

        if save_oof_folds:
            arch_img = f"{arch}_{img_size_opt}"
            oof_fold_path = os.path.join(oof_dir, f"{arch_img}_oof_fold{fold}.csv")
            oof_fold_df.to_csv(oof_fold_path, index=False)
            print(f"[OOF] Saved fold OOF: {oof_fold_path}")
        oof_chunks.append(oof_fold_df)
        
        if wrun is not None:
            try:
                wrun.finish()
            except Exception:
                pass
        
        del model, optimizer, scheduler, ema, train_loader, valid_loader, valid_loader_robust, train_ds, valid_ds, valid_ds_robust

        gc.collect()
        torch.cuda.empty_cache()

    summary_path = os.path.join(arch_dir, "train_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(fold_logs, f, ensure_ascii=False, indent=2)

    if oof_chunks:
        oof_df = pd.concat(oof_chunks, axis=0, ignore_index=True)
        oof_df = oof_df.sort_values("orig_idx").drop(columns=["orig_idx"])
        arch_img = f"{arch}_{img_size_opt}"
        oof_out = os.path.join(oof_dir, f"{arch_img}_oof.csv") 
        oof_df.to_csv(oof_out, index=False)
        print(f"[OOF] Saved merged: {oof_out} (shape={oof_df.shape})")

        if wandb_enabled:
            try:
                art = wandb.Artifact(f"{arch_img}_oof", type="oof")
                art.add_file(oof_out)
                wandb.log_artifact(art)
            except Exception:
                pass

    print("\nTraining finished.")
    print("Summary:", fold_logs)
    print(f"- Checkpoints saved under:  {arch_dir}")
    print(f"- Summary saved: {summary_path}")

    if wandb_enabled:
        try:
            art = wandb.Artifact("train_summary", type="summary")
            art.add_file(summary_path)
            wandb.log_artifact(art)
        except Exception:
            pass

    return summary_path