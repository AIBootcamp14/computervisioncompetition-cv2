import os
import json
import argparse
import pandas as pd
import torch

from src.train.train import run_train, list_models
from src.inference.inference import run_inference

try:
    import wandb
except Exception:
    wandb = None


def parse_minor_classes(s: str) -> tuple[int, ...]:
    s = s.strip()
    if not s:
        return tuple()
    return tuple(int(x) for x in s.split(",") if x.strip() != "")


def parse_focal_weight(s):
    if s is None:
        return None
    s = s.strip()
    if s == "" or s.lower() in {"none", "null", "false"}:
        return None
    if s.lower() in {"alpha_cb", "cb"}:
        return "alpha_cb"
    try:
        val = json.loads(s)
        if isinstance(val, list):
            return [float(x) for x in val]
        return val
    except Exception:
        pass
    try:
        return [float(x) for x in s.split(",") if x.strip() != ""]
    except Exception:
        return s


def parse_doc_heavy_probs(s):
    if not s or s.strip() == "":
        return {}
    
    s = s.replace(" ", "")
    
    result = {}
    for pair in s.split(","):
        if ":" not in pair:
            continue
        cls, prob = pair.split(":")
        result[int(cls)] = float(prob)
    return result


def main():
    """
    CLI 예시

    1) 학습 + 2-Stage 추론 + 클래스별 확률 저장
    python -m src.main \
      --arch convnextv2_base --epochs 70 --batch_size 32 --img_size auto \
      --tta --do_infer --save_proba \
      --stage1_presets "none,doc_deskew" \
      --stage2_preset doc_restore_bal \
      --stage2_mode entropy --route_gate 0.60 --stage2_blend 1.0

    2) timm 모델 목록 조회
    python -m src.main --list_models resnet
    """

    ap = argparse.ArgumentParser(description="Train & Ensemble Inference entry")

    ap.add_argument("--arch", type=str, default="resnet50", help="timm model (e.g., resnet50)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--minor_classes", type=str, default="1,13,14", help='소수 클래스 목록을 콤마로: "1,13,14"')
    ap.add_argument("--p_heavy_minor", type=float, default=0.7)
    ap.add_argument("--doc_heavy_probs", type=str, default="3:0.9,7:0.9,11:0.8", help="DOC classes heavy augmentation probabilities (format: 'class:prob,class:prob')")

    ap.add_argument("--no_early_stop", action="store_true", help="disable early stopping")
    ap.add_argument("--es_mode", type=str, default="either", choices=["loss", "f1", "both", "either"], help="Early stopping mode")
    ap.add_argument("--patience", type=int, default=10, help="early stopping patience (epochs)")
    ap.add_argument("--min_delta_loss", type=float, default=0.001, help="required LOSS improvement to reset patience")
    ap.add_argument("--min_delta_f1", type=float, default=0.0005, help="Minimum F1 improvement for early stopping")
    ap.add_argument("--eval_mode", type=str, default="f1", choices=["f1", "loss"], help="use checkpoint based on f1 or loss")
    ap.add_argument("--clip_value", type=float, default=2.0, help="Base gradient clipping value (auto-adjusted for large models)")
    ap.add_argument("--img_size", type=str, default="auto", help='image size: "auto"(default), "auto-long"(pad to square using longer side), or int(e.g., "512")')
    ap.add_argument("--accum_steps", type=int, default=1, help="gradient accumulation steps (e.g., accum_steps=4 with batch_size=8 gives effective 32)")

    ap.add_argument("--use_focal", action="store_true", help="use FocalLoss instead of LabelSmoothingCE")
    ap.add_argument("--focal_gamma", type=float, default=1.5, help="FocalLoss gamma (recommend 1.5~2.0)")
    ap.add_argument("--focal_weight", type=str, default=None, help='FocalLoss class weight: "alpha_cb" | JSON list | comma list | none')

    ap.add_argument("--use_logit_adjustment", action="store_true", help="subtract tau*log(class_prior) from logits during train/valid")
    ap.add_argument("--logit_tau", type=float, default=1.0, help="temperature for logit adjustment (tau)")

    ap.add_argument("--save_oof_folds", action="store_true", help="save per-fold OOF CSVs under output/oof/")
    ap.add_argument("--save_fold_logs", action="store_true", help="save per-fold train logs as CSV")

    ap.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases entity (e.g., team/org name)")
    ap.add_argument("--wandb_project", type=str, default=None, help="Weights & Biases project name")
    ap.add_argument("--wandb_run_name", type=str, default=None, help="Weights & Biases run name")

    ap.add_argument("--no_stage2", action="store_true", help="2단계 추론 비활성화")
    ap.add_argument("--do_infer", action="store_true", help="run inference after training")
    ap.add_argument("--save_proba", action="store_true", help="save per-class probabilities to CSV alongside predictions")
    ap.add_argument("--avg", type=str, default="logit", choices=["logit", "prob"], help='inference fold ensemble averaging: "logit" (default) or "prob"')
    ap.add_argument("--infer_batch_size", type=int, default=32)
    ap.add_argument("--tta", action="store_true", help="enable TTA (flip + 90/180/270 rotations)")
    ap.add_argument("--out_tag", type=str, default="", help="추론 출력 파일명 뒤에 붙일 태그")
    ap.add_argument("--stage1_presets", type=str, default="none,doc_deskew", help='1단계 추론시 적용할 증강: none, doc_deskew')
    ap.add_argument("--stage2_preset", type=str, default="doc_restore_bal", help='2단계 추론시 적용할 증강: doc_restore_bal | doc_restore_max')
    ap.add_argument("--stage2_mode", type=str, default="entropy", choices=["entropy", "margin", "sum", "pred"], help="2단계로 전달할 기준: entropy(default) | margin | sum | pred")
    ap.add_argument("--route_classes", type=str, default="", help="2단계 추론을 통해 엄밀한 구분을 하고 싶은 클래스 인덱스(콤마구분). stage2_mode 가 sum 혹은 pred일 때만 사용")
    ap.add_argument("--route_gate", type=float, default=0.60, help="2단계 추론 여부 결정의 임계값. entropy: 정규화 엔트로피>=gate, margin: (1 - (p1-p2))>=gate, sum: 선택클래스 합확률>=gate")
    ap.add_argument("--stage2_limit", type=int, default=0, help="2단계 추론을 진행할 샘플 수(0=제한 없음)")
    ap.add_argument("--stage2_blend", type=float, default=1.0, help="2단계 확률을 1단계와 블렌딩(1.0=교체, 0.5=절반/절반, 0=2단계 추론 미사용)")

    ap.add_argument("--list_models", type=str, default=None, help='search timm model (e.g. "--list_models resnet")')

    ap.add_argument("--root_dir", type=str, default=".")
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--model_dir", type=str, default="model")
    ap.add_argument("--output_dir", type=str, default="output")
    ap.add_argument("--test_dir", type=str, default="data/test")

    args = ap.parse_args()

    if args.list_models is not None:
        if args.list_models.lower() == "all":
            names = list_models(pretrained=True)
        else:
            names = list_models(filter_name=args.list_models, pretrained=True)
        print("\n".join(names))
        return

    root_dir = os.path.abspath(args.root_dir)
    data_dir = os.path.join(root_dir, args.data_dir)
    model_dir = os.path.join(root_dir, args.model_dir)
    output_dir = os.path.join(root_dir, args.output_dir)
    test_dir = os.path.join(root_dir, args.test_dir)
    train_csv_path = os.path.join(data_dir, "train.csv")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "oof"), exist_ok=True)

    df = pd.read_csv(train_csv_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    minor_classes = parse_minor_classes(args.minor_classes)
    doc_heavy_probs = parse_doc_heavy_probs(args.doc_heavy_probs)

    focal_weight = parse_focal_weight(args.focal_weight)

    summary_path = run_train(
        df=df,
        device=device,
        model_dir=model_dir,
        output_dir=output_dir,
        arch=args.arch,
        num_classes=17,
        img_size_opt=args.img_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_folds=args.n_folds,
        minor_classes=minor_classes,
        p_heavy_minor=args.p_heavy_minor,
        doc_heavy_probs=doc_heavy_probs,
        num_workers=args.num_workers,
        save_fold_logs=args.save_fold_logs,
        use_early_stopping=not args.no_early_stop,
        early_stopping_patience=args.patience,
        early_stopping_mode=args.es_mode,
        early_stopping_min_delta_loss=args.min_delta_loss,
        early_stopping_min_delta_f1=args.min_delta_f1,
        eval_mode=args.eval_mode,
        accum_steps=args.accum_steps,
        base_clip_value=args.clip_value,
        use_focal=args.use_focal,
        focal_gamma=args.focal_gamma,
        focal_weight=focal_weight, 
        use_logit_adjustment=args.use_logit_adjustment,
        logit_tau=args.logit_tau,
        save_oof_folds=args.save_oof_folds,
        wandb_entity=args.wandb_entity or os.environ.get("WANDB_ENTITY"),
        wandb_project=args.wandb_project or os.environ.get("WANDB_PROJECT"),
        wandb_run_name=args.wandb_run_name or os.environ.get("WANDB_NAME")
    )

    if args.do_infer:
        infer_args = {
            "arch": args.arch,
            "test_dir": test_dir,
            "summary_path": summary_path,
            "output_dir": output_dir,
            "img_size": args.img_size,
            "infer_batch_size": args.infer_batch_size,
            "num_workers": args.num_workers,
            "tta": args.tta,
            "save_proba": args.save_proba,
            "select_ckpt": args.eval_mode,
            "avg": args.avg,
            "out_tag": args.out_tag, 
            "stage1_presets": args.stage1_presets,
            "stage2_preset": args.stage2_preset,
            "stage2_mode": args.stage2_mode,
            "route_gate": args.route_gate,
            "route_classes": args.route_classes,
            "stage2_limit": args.stage2_limit,
            "stage2_blend": args.stage2_blend,
            "no_stage2": args.no_stage2,
        }
        run_inference(infer_args)

    if wandb is not None and getattr(wandb, "run", None) is not None:
        try:
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
