import os
import sys
import argparse
import pandas as pd
import torch

from src.train.train import run_train, list_models
from src.inference.inference import run_inference


def parse_minor_classes(s: str) -> tuple[int, ...]:
    """
    "1,13,14" 같은 문자열을 (1,13,14) 튜플로 변환
    """
    s = s.strip()
    if not s:
        return tuple()
    return tuple(int(x) for x in s.split(",") if x.strip() != "")


def main():
    ap = argparse.ArgumentParser(description="Train & Ensemble Inference entry")
    # 모델/학습 파라미터
    ap.add_argument("--arch", type=str, default="resnet50", help="timm model (e.g. resnet50)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--infer_batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--minor_classes", type=str, default="1,13,14",
                    help='소수 클래스 목록을 콤마로: "1,13,14"')
    ap.add_argument("--p_heavy_minor", type=float, default=0.7)
    ap.add_argument("--tta", action="store_true", help="enable simple TTA (flip/rot90)")

    # 실행 옵션
    ap.add_argument("--no_early_stop", action="store_true", help="disable early stopping")
    ap.add_argument("--patience", type=int, default=5, help="early stopping patience (epochs)")
    ap.add_argument("--min_delta", type=float, default=0.001, help="required f1 improvement to reset patience")
    ap.add_argument("--img_size", type=str, default="auto", help='image size: "auto"(default), "auto-long"(padded to square using longer side)), or int(e.g., "512")')
    ap.add_argument("--do_infer", action="store_true", help="run inference after training")
    ap.add_argument("--list_models", type=str, default=None, help='search timm model (e.g. "--list_models resnet")')
    ap.add_argument("--save_proba", action="store_true", help="save per-class probabilities to CSV alongside predictions")

    # 경로 옵션(필요시 변경 가능)
    ap.add_argument("--root_dir", type=str, default=".")
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--model_dir", type=str, default="model")
    ap.add_argument("--output_dir", type=str, default="output")
    ap.add_argument("--test_dir", type=str, default="data/test")

    args = ap.parse_args()

    # timm 모델 리스트 요청
    if args.list_models is not None:
        if args.list_models.lower() == "all":
            names = list_models(pretrained=True)
        else:
            names = list_models(filter_name=args.list_models, pretrained=True)
        print("\n".join(names))
        return 

    # 경로 설정
    root_dir = os.path.abspath(args.root_dir)
    data_dir = os.path.join(root_dir, args.data_dir)
    model_dir = os.path.join(root_dir, args.model_dir)
    output_dir = os.path.join(root_dir, args.output_dir)
    test_dir = os.path.join(root_dir, args.test_dir)
    train_csv_path = os.path.join(data_dir, "train.csv")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # 데이터 불러오기 (train.csv: ID, target 컬럼 가정)
    df = pd.read_csv(train_csv_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 소수 클래스 정의
    minor_classes = parse_minor_classes(args.minor_classes)

    # fold별 체크포인트 + train_summary 저장
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
        num_workers=args.num_workers,
        save_fold_logs=False,  # csv 파일로 로그 필요시 True
        use_early_stopping=not args.no_early_stop,
        early_stopping_patience=args.patience,
        early_stopping_min_delta=args.min_delta
    )

    # 앙상블 추론
    if args.do_infer:
        sys_argv = [
            "inference",
            "--arch", args.arch,
            "--test_dir", test_dir,
            "--summary_path", summary_path,     # model/{arch}/train_summary.json
            "--output_dir", output_dir,
            "--infer_batch_size", str(args.infer_batch_size),
            "--num_workers", str(args.num_workers),
            "--save_proba"
        ]
        if getattr(args, "tta", False):
            sys_argv.append("--tta")

        sys.argv = sys_argv
        run_inference()


if __name__ == "__main__":
    main()
