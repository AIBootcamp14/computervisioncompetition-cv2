import os
import sys
import argparse
import pandas as pd
import torch
import timm

from src.train.train import run_train
from src.inference.inference import run_inference

def list_models(filter_name=None, pretrained=True):
    """
    timm에서 사용 가능한 모델 목록을 반환합니다.
    """
    model_list = timm.list_models(filter_name, pretrained)
    return model_list

def parse_minor_classes(minor_classes_str):
    """
    콤마로 구분된 문자열을 정수 리스트로 변환합니다.
    """
    return [int(c) for c in minor_classes_str.split(',')]

def main():
    """    CLI 명령어 예시        예시 1)    python -m src.main --arch resnet50 --epochs 50 --batch_size 32 --do_infer --tta        * 이용하는 모델: timm에 pretrain 모델 중 resnet50    * 학습 파라미터 설정:             - epochs = 50             - batch_size = 32        * 나머지는 default:               - lr = 3e-4            - n_folds = 5               (stratified k fold)            - num_workers = 4            - minor_classes = (1,13,14) (데이터가 적은 class)            - p_heavy_minor = 0.7       (데이터가 적은 class의 데이터를 얼마나 더 심하게 변형하여 다양하게 학습시킬 것인지 확률 설정)    * 학습시 실행 옵션은 default:            - no_early_stop = False     (early stopping 이용,                - patience = 5           5번의 epoch동안                - min_delta = 0.001      valid f1 score가 0.001 이상 증가하지 않으면 학습 종료)            - img_size = auto           (이미지 사이즈 설정은 resnet50의 설정을 따라감)    * 추론시 실행 옵션 설정:            - do_infer = True           (학습 종료 후 추론까지 진행)        * 나머지는 default:            - save_proba = False        (각 class별 확률값은 output으로 저장하지 않음)    * 추론 파라미터 설정:               - tta = True                (test 이미지를 회전 및 flip 변형한 것에 대한 추론도 병행 → 추론 결과 확률을 모두 더해 가장 가능성이 높은 class로 분류)        * 나머지는 default:            - infer_batch_size = 64     (추론시 batch_size = 64)    * 경로 설정은 default: 아래 코드의 default 경로에 모델 및 결과물이 저장됨    예시 2)    python -m src.main --arch convnextv2_base --epochs 50 --batch_size 32 --do_infer --tta --save_proba    예시 3) timm에서 쓸 수 있는 모델 목록을 보고 싶은 경우    python -m src.main --list_models            (모든 사용가능한 모델 목록을 보여줌)    python -m src.main --list_models resnet     (resnet 이름이 들어간 모든 모델 목록을 보여줌)    """
    ap = argparse.ArgumentParser(description="Train & Ensemble Inference entry")
    # 모델/학습 파라미터
    ap.add_argument("--arch", type=str, default="resnet50", help="timm model (e.g., resnet50)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--minor_classes", type=str, default="1,13,14",
                    help='소수 클래스 목록을 콤마로: "1,13,14"')
    ap.add_argument("--p_heavy_minor", type=float, default=0.7)

    # 학습시 실행 옵션
    ## early stopping 기능
    ap.add_argument("--no_early_stop", action="store_true", help="disable early stopping")
    ap.add_argument("--patience", type=int, default=5, help="early stopping patience (epochs)")
    ap.add_argument("--min_delta", type=float, default=0.001, help="required f1 improvement to reset patience")
    ## img size 를 어떻게 맞출 것인지 설정 
    # auto(default): 사용하는 pretrain 모델의 설정을 따라감
    # auto-long: 사진의 긴 변을 기준으로 정사각형으로 설정 (추가된 부분은 흰색으로 패딩)
    # 정수 지정: 지정한 정수의 크기를 한 변으로 하는 정사각형의 사이즈로 설정 (추가된 부분은 흰색으로 패딩)
    ap.add_argument("--img_size", type=str, default="auto", help='image size: "auto"(default), "auto-long"(padded to square using longer side)), or int(e.g., "512")')

    # 추론시 실행 옵션
    ## 학습 후 추론까지 실행할 것인지 여부 설정
    ap.add_argument("--do_infer", action="store_true", help="run inference after training")
    ## 각 class별 확률까지 저장할 것인지 여부 설정 (ensemble에 모델을 이용한다면 필요)
    ap.add_argument("--save_proba", action="store_true", help="save per-class probabilities to CSV alongside predictions")

    # 추론 파라미터
    ap.add_argument("--infer_batch_size", type=int, default=64)
    ap.add_argument("--tta", action="store_true", help="enable simple TTA (flip/rot90/180/270)")

    # 기타 실행 옵션
    ap.add_argument("--list_models", type=str, default=None, help='search timm model (e.g. "--list_models resnet")')

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