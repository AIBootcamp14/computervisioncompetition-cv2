# Document Type Classification

이 프로젝트는 문서 이미지 분류 대회 참가에 사용된 솔루션으로, 컴퓨터 비전 딥러닝 기술들을 효과적으로 조합하여 최고 성능을 달성했습니다. 

특히, 모델의 일반화 성능을 극대화하고 최종 예측 정확도를 끌어올리기 위한 앙상블 및 스태킹 기법에 중점을 두어 개발되었습니다.

## Team

| ![김장원](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김영](https://avatars.githubusercontent.com/u/156163982?v=4) | ![민병호](https://avatars.githubusercontent.com/u/156163982?v=4) | ![문채린](https://avatars.githubusercontent.com/u/156163982?v=4) | ![정민지](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [김장원](https://github.com/jkim1209)             |            [김영](https://github.com/kimyoung9689)             |            [민병호](https://github.com/BH-Min-lab)             |            [문채린](https://github.com/CHAERINMOON)             |            [정민지](https://github.com/mingg210)             |
|                            팀장, 데이터 전처리 및 모델링                             |                            데이터 전처리 및 모델링                             |                            데이터 전처리 및 VLM                             |                            데이터 전처리 및 모델링                             |                            데이터 전처리 및 모델링                             |

## 0. Overview

모델 아키텍처: Vision Transformer (ViT) 기반 vit_base_patch16_224

학습 전략: K-Fold Cross Validation

손실 함수: Focal Loss with Class Balancing, Logit Adjustment

최종 예측: Stacking Ensemble (Logistic Regression)

환경: 
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![timm](https://img.shields.io/badge/Timm-007396?style=for-the-badge&logo=Timm&logoColor=white)

최종 점수: Test Macro F1 Score 0.97 달성

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




```cmd
├── data/                          # 데이터(.csv)
│   ├── test/                      # 테스트 이미지(.jpg)
│   └── train/                     # 학습 이미지(.jpg)
├── model/                        # 학습된 모델 체크포인트(.pt) 및 학습 요약(.json) 저장
│   ├── vit_base_patch16_224/     # 각 모델별로 체크포인트가 저장
│   │   ├── fold0_best_f1_light.pt
│   │   └── train_summary.json     # 전체 학습 및 검증 결과 요약
│   └── ...
├── output/                        # 최종 예측 및 중간 결과물 저장
│   ├── oof/                       # Cross-Validation OOF(Out-of-Fold) 예측 파일
│   └── submission/                # 최종 제출용 예측 파일
├── src/                          # 핵심 소스 코드
│   ├── data/                      # 데이터 전처리 및 로더 관련
│   ├── evaluate/                  # 모델 평가 지표 관련
│   ├── inference/                 # 모델 추론 로직
│   └── train/                     # 모델 학습 로직
├── .gitignore                      # 버전 관리 제외 파일 목록
├── environment.yml                # Conda 환경 설정 파일
├── ensemble.py                    # 스태킹 앙상블 실행 스크립트
├── main.py                        # 전체 파이프라인 실행 스크립트
└── output_to_submission.py        # 최종 제출 파일 변환 유틸리티
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

### Model descrition

-모델 선정: 대회 데이터셋은 실제 문서 이미지로, 복잡한 패턴과 텍스트가 포함되어 있습니다. Vision Transformer (ViT)는 이미지의 전역적인 특징을 효과적으로 학습하는 데 강점이 있어 이 문제에 적합하다고 판단했습니다.

### Final Prediction: Ensemble & Stacking
K-Fold 교차 검증: 모델의 일반화 성능을 확인하고 최종 예측의 신뢰도를 높이기 위해 5-Fold 교차 검증을 사용했습니다.

손실 함수: 클래스 불균형 문제를 해결하기 위해 Focal Loss와 Logit Adjustment를 적용했습니다.

앙상블 및 스태킹: 개별 모델의 예측 결과를 합쳐 최종 성능을 극대화하는 스태킹 앙상블을 사용했습니다. 이는 ensemble.py 스크립트를 통해 구현되었습니다.

## 5. Result

### Leader Board

- 최종 순위: []위
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