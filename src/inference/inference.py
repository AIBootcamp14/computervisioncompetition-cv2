import os
import sys
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
import json
import timm

from src.data.dataset import ImgDataset
from src.data.transform import build_test_tf
from src.train.train import run_train
from src.utils.utils import get_model, setup_seed
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm

def get_config_from_summary(summary_path):
    with open(summary_path, 'r') as f:
        config = json.load(f)
    return config

def run_inference():
    ap = argparse.ArgumentParser(description="Ensemble Inference entry")
    ap.add_argument("--arch", type=str, default="resnet50", help="timm model (e.g., resnet50)")
    ap.add_argument("--summary_path", type=str, required=True, help="Path to the training summary JSON file")
    ap.add_argument("--test_dir", type=str, default="data/test", help="Path to the test image directory")
    ap.add_argument("--output_dir", type=str, default="output", help="Path to save output files")
    ap.add_argument("--infer_batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--save_proba", action="store_true", help="Save per-class probabilities")
    ap.add_argument("--tta", action="store_true", help="Enable TTA")
    
    args, _ = ap.parse_known_args()

    config = get_config_from_summary(args.summary_path)
    model_name = config['arch']
    n_folds = config['n_folds']
    num_classes = 17 # Assuming 17 classes from context

    setup_seed(42)

    submission_df = pd.read_csv(os.path.join(args.test_dir, '..', 'sample_submission.csv'))
    test_files = [os.path.join(args.test_dir, f) for f in submission_df['ID']]
    test_df = pd.DataFrame({'file_path': test_files})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_fold_preds = []
    
    for fold in range(n_folds):
        print(f"\n--- Inference Fold {fold+1}/{n_folds} ---")
        
        model = get_model(model_name, num_classes).to(device)
        model_path = os.path.join(os.path.dirname(args.summary_path), f"best_model_fold_{fold}.pt")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        if args.tta:
            test_tf = build_test_tf(tta=True)
        else:
            test_tf = build_test_tf(tta=False)
        
        test_ds = ImgDataset(test_df, is_train=False)
        test_loader = DataLoader(
            test_ds,
            batch_size=args.infer_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

        fold_preds = []
        with torch.no_grad():
            for images, _ in tqdm(test_loader, desc=f"Fold {fold+1} Inference"):
                images = images.to(device)
                outputs = model(images)
                fold_preds.append(outputs.cpu().numpy())

        all_fold_preds.append(np.concatenate(fold_preds, axis=0))

    avg_proba = np.mean(all_fold_preds, axis=0)
    final_preds = np.argmax(avg_proba, axis=1)

    submission_df['target'] = final_preds
    submission_df.to_csv(os.path.join(args.output_dir, f"{model_name}_ensemble_submission.csv"), index=False)

    print("\nInference complete. Submission file saved.")

if __name__ == "__main__":
    run_inference()