# Document Type Classification

이 프로젝트는 문서 이미지 분류 대회 참가에 사용된 솔루션으로, 컴퓨터 비전 딥러닝 기술들을 효과적으로 조합하여 최고 성능을 달성했습니다. 

특히, 모델의 일반화 성능을 극대화하고 최종 예측 정확도를 끌어올리기 위한 앙상블 및 스태킹 기법에 중점을 두어 개발되었습니다.

## 2조 팀원 소개

| ![김장원](https://avatars.githubusercontent.com/u/128503571?v=4&s=200) | ![김영](https://avatars.githubusercontent.com/u/156163982?v=4) | ![민병호](https://avatars.githubusercontent.com/u/156163982?v=4) | ![문채린](https://avatars.githubusercontent.com/u/156163982?v=4) | ![정민지](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            ![GitHub](https://img.shields.io/badge/GitHub-김장원👑-181717?style=&logo=github&logoColor=white&link=https://github.com/jkim1209)             |            [![GitHub](https://img.shields.io/badge/GitHub-김영-181717?style=flat&logo=github&logoColor=white)](https://github.com/kimyoung9689)            |            [![GitHub](https://img.shields.io/badge/GitHub-민병호-181717?style=flat&logo=github&logoColor=white)](https://github.com/BH-Min-lab)            |            [![GitHub](https://img.shields.io/badge/GitHub-문채린-181717?style=flat&logo=github&logoColor=white)](https://github.com/CHAERINMOON)             |            [![GitHub](https://img.shields.io/badge/GitHub-정민지-181717?style=flat&logo=github&logoColor=white)](https://github.com/mingg210)          |
|                            팀장, 데이터 전처리 및 모델링                             |                            데이터 전처리 및 모델링                             |                            데이터 전처리 및 VLM                             |                            데이터 전처리 및 모델링                             |                            데이터 전처리 및 모델링                             |

## 0. Overview

모델 아키텍처: Vision Transformer (ViT) 기반 vit_base_patch16_224

학습 전략: 클래스별 맞춤 데이터 증강과 불균형 보정, CosineAnnealingWarmup 스케줄러 기반 최적화, 엄격한 valid set 검증 및 2-Stage 라우팅 추론을 결합한 학습

손실 함수: Focal Loss with Class Balancing

최종 점수: Test Macro F1 Score 0.97 달성

환경: 
![Python](https://img.shields.io/badge/Python-3776AB?style=plastic&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=plastic&logo=pytorch&logoColor=white)
![Torchvision](https://img.shields.io/badge/Torchvision-EE4C2C?style=plastic&logo=pytorch&logoColor=white)
![timm](https://img.shields.io/badge/timm-EE4C2C?style=plastic&logo=pytorch&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=plastic&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=plastic&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-0B5FFF?style=plastic&logo=numpy&logoColor=white)
![Albumentations](https://img.shields.io/badge/Albumentations-FF6F00?style=plastic)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=plastic&logo=opencv&logoColor=white)
![W&B](https://img.shields.io/badge/Weights%20%26%20Biases-FFBE00?style=plastic&logo=weightsandbiases&logoColor=black)


### Environment

- OS: Ubuntu 20.04.6 LTS (tested)  
- Python: 3.10  
- PyTorch: 2.1.0  
- CUDA: 11.8, cuDNN 8.7  
- GPU: NVIDIA GeForce RTX 3090 (24GB VRAM)  
- Dependencies: See [environment.yml](./environment.yml)

### Requirements

- python=3.10
- pytorch=2.1.0
- pytorch-cuda=11.8
- torchvision=0.16.0
- torchaudio=2.1.0
- numpy=1.26
- scipy=1.11
- scikit-learn=1.3
- pandas=2.1


## 1. Competiton Info

### Overview

이 대회는 문서 타입 분류를 위한 이미지 분류 대회로, 금융, 의료, 보험, 물류 등 다양한 산업 분야에서 활용되는 문서 이미지를 17개 클래스로 분류합니다. 현업 데이터를 기반으로 제작되어 실제 문제를 해결하는 경험을 제공합니다.

![alt text](image.png)



- **기간**: 2025년 9월 1일 ~ 2025년 9월 11일

- **주제**: 문서 이미지 분류

- **평가지표**: Macro F1 Score

$$
\text{Macro-F1} = \frac{1}{N} \sum_{i=1}^{N} F1_i
$$

$$
F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}
$$

 > - $N$: 클래스 개수 (이번 대회의 경우 17)  
 > - $F1_i$: i번째 클래스의 F1 Score  
 > - Precision = $\frac{TP}{TP + FP}$  
 > - Recall = $\frac{TP}{TP + FN}$  
 > - TP: True Positive, FP: False Positive, FN: False Negative  

- **주요 목표**:
  - **다양한 CNN 기반 Backbone (ResNet, EfficientNet 등)** 및 최신 기법을 적용 및 성능 비교  
  - **데이터 불균형 문제를 고려한 학습 기법 실험**(데이터 증강, 클래스 가중치 적용 등) 실험  
  - **Macro-F1 기준 최적 성능 달성**을 목표로 하는 모델 구축

## 2. Components

### Directory

프로젝트는 모듈성과 재사용성을 극대화하기 위해 다음과 같은 명확한 디렉토리 구조를 따릅니다.


```txt
.
├── data
│   ├── noisy               # src/data/EDA_mismatch.py 로 확인한 잘못 예측한 이미지 파일 저장
│   ├── test                # test 이미지 파일 저장
│   └── train               # train 이미지 파일 저장
├── img                     # EDA 결과 및 필요한 이미지 파일 저장
├── logs                    # run_scripts 배쉬 파일 실행 로그 저장
├── model                   # 각 모델 디렉토리 안에 fold별 .pt 파일 및 train_summary.json 저장
│   ├── convnext_base.fb_in22k_ft_in1k_384_auto     
│   ├── efficientnet_b4.ra2_in1k_auto
│   ├── maxvit_base_tf_512.in21k_ft_in1k_auto
│   ├── tf_efficientnet_b7.ap_in1k_auto
│   ├── vit_base_patch14_dinov2.lvd142m_auto
│   ├── vit_base_patch16_siglip_512.webli_auto
│   └── ...
├── output                  # 결과물 .csv 파일 저장
│   └── oof                 # OOF .csv 파일 저장
├── run_scripts             # 실행 명령어 .sh 파일 저장
├── src                     # 스크립트 .py 파일 저장
│   ├── data
│   │   ├── EDA.py
│   │   ├── EDA_mismatch.py
│   │   ├── dataset.py
│   │   └── transform.py
│   ├── ensemble.py
│   ├── evaluate
│   │   ├── evaluate.py
│   │   └── identify_possible_mislabeled_data.py
│   ├── inference
│   │   └── inference.py
│   ├── main.py
│   ├── output_to_submission.py
│   └── train
│       ├── correct_minor_classes.py
│       ├── schedulers.py
│       └── train.py
└── wandb                   # wandb 파일 저장
    ├── latest-run
    └── ...
```


## 3. Data descrption

### Dataset overview

- 이번 대회 데이터는 금융, 의료, 물류 등 다양한 분야의 실제 문서 이미지를 기반으로 구축되었으며, 총 17개의 클래스로 분류됩니다. 학습 데이터 1,570장, 평가 데이터 3,140장으로 구성되어 있습니다.

### EDA

- 데이터 탐색(EDA)을 통해 다음과 같은 특징을 발견했습니다.

클래스 불균형: 특정 클래스(1, 13, 14)의 데이터 수가 매우 적은 불균형 문제를 확인했습니다.

이미지 해상도: 이미지 크기는 512px에서 763px까지 다양하게 분포되어 있음을 파악했습니다.

레이블 정확도: 일부 오분류된 이미지들이 존재하여, 데이터 레이블에 휴먼 에러가 일부 발견되었습니다.

### Data Processing

전처리: 모든 이미지를 모델의 입력 크기에 맞게 리사이즈하고, RGB 채널별로 평균 및 표준 편차를 사용해 정규화했습니다.

데이터 증강: 학습 데이터의 일반화 성능을 높이기 위해 무작위 회전, 상하좌우 반전 등 다양한 증강 기법을 적용했습니다.

## 4. Modeling

### Model description

- **모델 & 입력 크기 자동화**  
  - `timm` 기반 아키텍처 사용. 입력 크기는 `"auto"`, `"auto-long"`, 정수 지정 가능.  
  - 지원 모델 예시: ConvNeXt, EfficientNet, MaxViT, ViT(DINOv2, SigLIP) 등.

- **데이터 분할 & 재현성**  
  - `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` 사용.  
  - 시드 고정 및 torch·numpy·cudnn의 deterministic 모드 적용.

- **데이터 증강 (클래스별 파이프라인)**  
  - 문서류(DOC), 신분증/여권(ID_CERT), 차량(CAR) 세 집합으로 분리.  
  - 문서류는 OverlayFromPool 기반 **self/pool 혼합 overlay** 추가.  
  - 소수 클래스는 heavy transform 확률적 적용.

- **손실함수 & 최적화**  
  - 기본: LabelSmoothingCE  
  - 옵션: FocalLoss(γ=1.5~2.0, class-balanced weight 지원).  
  - Mixed Precision, Accumulation, Gradient Clipping, EMA 적용.  
  - 스케줄러: CosineAnnealingWarmupRestarts 적용.

- **Early Stopping & 체크포인트**  
  - `es_mode={loss|f1|both|either}`, patience=10, Δloss=0.001, Δf1=0.0005.  
  - fold별 `*_best_loss.pt`, `*_best_f1.pt` 저장.

- **검증 체계 (Validation)**  
  - clean valid + robust valid (노이즈, 압축, 다운스케일 캐시 고정).  
  - robust 검증은 실세계 성능 근접 모니터링 지표로 사용.

- **추론 (Inference)**  
  - Fold 앙상블 평균(logit or prob).  
  - TTA 지원(Flip, 90/180/270 회전).  
  - Stage-2 라우팅: 불확실 샘플만 `doc_restore_bal` 또는 `doc_restore_max` preset으로 재추론.

- **스태킹 앙상블 (Ensemble)**  
  - 다중 모델 OOF/Test 확률 → Logistic Regression meta-model.  
  - Feature 모드: `proba`, `logproba`, `both`. 내부 CV로 `C` 선택, 지표=macro F1.


#### 이미지 증강 파이프라인

| 클래스 | Base Transform | Heavy Transform | 추가 특징 |
|--------|----------------|-----------------|-----------|
| 문서류(DOC) | Affine, Rotate90, HueSatShift, CLAHE, Downscale, Noise | 더 강한 Affine/Noise, OverlayFromPool(base/heavy) | Document 전용 Overlay 추가 |
| 신분증/여권(ID) | Affine, ColorJitter, Noise (중간 강도) | 일부 heavy | OCR-like 손상 |
| 차량(CAR) | Affine, Contrast, Blur, Noise | 일부 heavy | 반사/노이즈 강화 |


#### 아키텍처 & 하이퍼파라미터 조합 예시

| 모델 아키텍처 | 입력 크기 | Epochs | Effective Batch Size | LR (max) | 스케줄러 | 손실함수 |
|---------------|-----------|--------|------------|-----------|----------|-----------|
| ConvNeXt-B    | auto(384) | 150    | 32         | 3e-4      | CosineAnnealingWarmupRestarts | LabelSmoothingCE |
| EfficientNet-B4 | 512     | 150     | 32         | 3e-4      | 동일     | LabelSmoothingCE |
| MaxViT-B      | auto(512)       | 150     | 32         | 1e-4      | 동일     | Focal |
| ViT-B SigLIP  | auto(512)       | 150     | 32         | 7e-5      | 동일     | Focal |
| ViT-B DINOv2  | auto(518)   | 150     | 32         | 5e-5      | 동일     | Focal|

```mermaid
flowchart TD
    A[Stage-1 Inference<br>Fold Ensemble + TTA] --> B{Route Decision}
    B -->|확실| C[Final Prediction<br>(Stage-1 prob)]
    B -->|불확실| D[Stage-2 Inference<br>Preset: doc_restore_bal/max]
    D --> E[Blend Stage-1 & Stage-2 Proba<br>(blend=0~1)]
    E --> F[Final Prediction]
```

#### 2단계 라우팅 추론 다이어그램

```mermaid
flowchart TD
    A[Stage-1 Inference<br>Fold Ensemble + TTA] --> B{Route Decision}
    B -->|확실| C[Final Prediction<br>(Stage-1 prob)]
    B -->|불확실| D[Stage-2 Inference<br>Preset: doc_restore_bal/max]
    D --> E[Blend Stage-1 & Stage-2 Proba<br>(blend=0~1)]
    E --> F[Final Prediction]
```

| 항목 | 설명 |
|------|------|
| Routing 기준 | entropy / margin / sum / pred |
| Gate | default = 0.60, Stage-2 샘플 수 제한(`stage2_limit`) 가능 |
| Blend | 1.0 = Stage-2만, 0.5 = 동등 블렌드, 0 = Stage-1 유지 |


### Final Prediction: Ensemble & Stacking
K-Fold 교차 검증: 모델의 일반화 성능을 확인하고 최종 예측의 신뢰도를 높이기 위해 5-Fold 교차 검증을 사용했습니다.

손실 함수: 클래스 불균형 문제를 해결하기 위해 Focal Loss와 Logit Adjustment를 적용했습니다.

앙상블 및 스태킹: 개별 모델의 예측 결과를 합쳐 최종 성능을 극대화하는 스태킹 앙상블을 사용했습니다. 이는 ensemble.py 스크립트를 통해 구현되었습니다.



## 5. Result

### Leader Board

- 최종 순위: 1위
- 최종 점수: [Test Macro F1 Score 0.9713]
이거 7시 이후 사진으로 넣을 것

여기에 EDA 단계에서 만든 클래스분포 그래프나 
케이 폴드 교차검증 학습/평증 곡선
or 혼동행렬그래프 넣어주면 좋을 거 같습니다.

예시
데이터셋 클래스 분포를 시각화한 그래프 (데이터 불균형을 한눈에 보여줌)

최종 성능 결과를 보여주는 Confusion Matrix



### Presentation

- 발표자료 링크

## etc

### Meeting Log

- [_Insert your meeting log link like Notion or Google Docs_](https://github.com/orgs/AIBootcamp14/projects/25)

### Reference

- [_Insert related reference_](https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup/blob/master/cosine_annealing_warmup/scheduler.py)




## 실행 가이드 




### Setup

#### 1. 환경 설정

```bash
conda env create -f environment.yml
conda activate base
```

#### 2. 학습 및 추론 실행
main.py를 통해 전체 파이프라인을 제어합니다. --do_infer 플래그를 추가하면 학습 후 자동으로 추론까지 실행됩니다.

```bash
python main.py \
    --arch vit_base_patch16_224 \
    --img_size 224 \
    --n_folds 5 \
    --use_focal \
    --do_infer \
    --save_oof_folds \
    --avg logit \
    --tta
```

--arch vit_base_patch16_224: timm 라이브러리의 ViT 기반 모델을 사용합니다.

--img_size 224: 이미지 크기를 224x224로 리사이즈합니다.

--n_folds 5: 5-Fold Cross Validation을 수행합니다.

--use_focal: Focal Loss를 사용합니다.

--do_infer: 학습이 완료된 후 추론을 실행합니다.

--save_oof_folds: 각 폴드별 OOF 예측 파일을 저장합니다.

--avg logit: 추론 시 앙상블 평균을 Logit 단위로 계산합니다.

--tta: 추론 시 TTA(Test Time Augmentation)를 적용합니다.

#### 3. 스태킹 앙상블 실행
main.py를 통해 생성된 OOF 파일들을 사용하여 스태킹 앙상블을 수행합니다. 이 과정은 개별 모델의 성능을 넘어 최종 예측 정확도를 극대화하는 핵심 단계입니다.

 `output/oof/`에 저장된 OOF 파일들을 인자로 전달
```bash
python ensemble.py \
    --oof output/oof/vit_base_patch16_224_oof.csv output/oof/swinv2_base_window12to24_192to384_oof.csv \
    --test output/test/vit_base_patch16_224_test.csv output/test/swinv2_base_window12to24_192to384_test.csv \
    --feature both \
    --standardize
```

--oof: 학습 과정에서 생성된 OOF CSV 파일 목록을 지정합니다.

--test: 각 모델의 테스트셋 예측 CSV 파일 목록을 지정합니다.

--feature both: 로지스틱 회귀 모델의 입력으로 확률값과 로그 확률값을 모두 사용합니다.

--standardize: 입력 특징을 표준화하여 안정적인 학습을 유도합니다.

#### 4. 최종 제출 파일 생성
스태킹 앙상블로 생성된 최종 예측 CSV를 대회 제출 형식에 맞게 변환합니다.

```bash
python output_to_submission.py
```