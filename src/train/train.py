import os
import sys
import json
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from datetime import datetime

from ..data.dataset import ImgDataset
from ..data.transform import build_train_tf_base, build_valid_tf, build_train_tf_heavy, build_test_tf
from .focal_loss import FocalLoss
from ..evaluate.evaluate import evaluate_macro_f1
from ..utils.utils import get_model, save_checkpoint, setup_seed, get_img_size_from_arch

def run_train(
    df,
    device,
    model_dir,
    output_dir,
    arch,
    num_classes,
    img_size_opt,
    epochs,
    batch_size,
    lr,
    n_folds,
    minor_classes,
    p_heavy_minor,
    num_workers,
    save_fold_logs,
    use_early_stopping,
    early_stopping_patience,
    early_stopping_min_delta
):
    model_name = arch
    seed = 42
    setup_seed(seed)
    
    img_size = get_img_size_from_arch(arch, img_size_opt)

    df["fold"] = -1
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (_, valid_idx) in enumerate(skf.split(df, df["target"])):
        df.loc[valid_idx, "fold"] = fold

    model_summary = {
        "arch": model_name,
        "n_folds": n_folds,
        "folds": {},
        "avg_valid_f1": 0.0,
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    fold_metrics = []
    
    criterion = FocalLoss(gamma=2, alpha=None)
    
    for fold in range(n_folds):
        print(f"\n--- Fold {fold+1}/{n_folds} 훈련 시작 ---")

        train_df = df[df["fold"] != fold].reset_index(drop=True)
        valid_df = df[df["fold"] == fold].reset_index(drop=True)
        
        train_ds = ImgDataset(train_df, is_train=True, transform_heavy_ratio=p_heavy_minor, minor_classes=minor_classes)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        valid_ds = ImgDataset(valid_df, is_train=False)
        valid_loader = DataLoader(
            valid_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        model = get_model(arch, num_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        best_f1 = 0.0
        best_epoch = -1
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            scheduler.step()

            valid_f1, _ = evaluate_macro_f1(model, valid_loader, device)
            
            print(f"Epoch [{epoch}/{epochs}], Loss: {epoch_loss:.4f}, Valid F1: {valid_f1:.4f}")

            if valid_f1 > best_f1:
                best_f1 = valid_f1
                best_epoch = epoch
                patience_counter = 0
                save_checkpoint(model, model_dir, model_name, fold, is_best=True)
            else:
                patience_counter += 1

            if use_early_stopping and patience_counter >= early_stopping_patience:
                print(f"Early stopping on fold {fold} at epoch {epoch}. Best F1: {best_f1:.4f}")
                break

        print(f"Fold {fold+1} Best F1 Score: {best_f1:.4f} at epoch {best_epoch}")
        fold_metrics.append({"fold": fold, "best_f1": best_f1, "best_epoch": best_epoch})
        
        model_summary["folds"][f"fold_{fold}"] = {
            "best_f1": best_f1,
            "best_epoch": best_epoch
        }

    avg_f1 = sum(m["best_f1"] for m in fold_metrics) / len(fold_metrics)
    model_summary["avg_valid_f1"] = avg_f1
    
    summary_path = os.path.join(model_dir, model_name, "train_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(model_summary, f, indent=4)
        
    print(f"\n모델 훈련 완료. 평균 Valid F1 Score: {avg_f1:.4f}")
    return summary_path