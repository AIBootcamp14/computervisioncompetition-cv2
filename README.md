# Document Type Classification

이 프로젝트는 문서 이미지 분류 대회 참가에 사용된 솔루션으로, 컴퓨터 비전 분야의 최신 딥러닝 기술들을 효과적으로 조합하여 최고 성능을 달성한 결과물입니다. 

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

이번 대회는 문서 타입 분류를 위한 이미지 분류 대회입니다. 문서 데이터는 금융, 의료, 보험, 물류 등 산업 전반에 가장 많은 데이터이며, 많은 대기업에서 디지털 혁신을 위해 문서 유형을 분류하고자 합니다. 이러한 문서 타입 분류는 의료, 금융 등 여러 비즈니스 분야에서 대량의 문서 이미지를 식별하고 자동화 처리를 가능케 할 수 있습니다.

이번 대회에 사용될 데이터는 총 17개 종의 문서로 분류되어 있습니다. 1570장의 학습 이미지를 통해 3140장의 평가 이미지를 예측하게 됩니다. 특히, 현업에서 사용하는 실 데이터를 기반으로 대회를 제작하여 대회와 현업의 갭을 최대한 줄였습니다. 또한 현업에서 생길 수 있는 여러 문서 상태에 대한 이미지를 구축하였습니다.

이번 대회를 통해서 문서 타입 데이터셋을 이용해 이미지 분류를 모델을 구축합니다. 주어진 문서 이미지를 입력 받아 17개의 클래스 중 정답을 예측하게 됩니다. computer vision에서 중요한 backbone 모델들을 실제 활용해보고, 좋은 성능을 가지는 모델을 개발할 수 있습니다. 그 밖에 학습했던 여러 테크닉들을 적용해 볼 수 있습니다.

- **기간**: 2025년 9월 1일 ~ 2025년 9월 11일

- **주제**: 문서 이미지 분류

- **평가지표**: Macro F1

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
  - **다양한 CNN 기반 Backbone (ResNet, EfficientNet 등)** 및 최신 기법을 적용해 성능 비교  
  - **데이터 불균형 문제를 고려한 학습 기법**(데이터 증강, 클래스 가중치 적용 등) 실험  
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

레이블 정확도: 일부 노이즈가 존재했지만, 레이블은 전반적으로 깨끗한 상태임을 확인했습니다.

### Data Processing

- 모델 학습을 위해 다음과 같은 데이터 전처리 및 증강 기법을 적용했습니다.

리사이즈 및 정규화: 모든 이미지를 모델의 입력 크기에 맞게 리사이즈하고, RGB 채널별로 평균 및 표준 편차를 사용해 정규화했습니다.

데이터 증강: 학습 데이터의 일반화 성능을 높이기 위해 무작위 회전, 상하좌우 반전 등 다양한 증강 기법을 적용했습니다.

## 4. Modeling

### Model descrition

- _Write model information and why your select this model_

### Modeling Process

See `run_scripts/`

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

1. 환경 설정

```bash
conda env create -f environment.yml
conda activate base
```

2. 학습 및 추론 실행
전체 파이프라인은 main.py를 통해 제어됩니다. --do_infer 플래그를 추가하면 학습 후 자동으로 추론까지 실행됩니다.

학습 및 추론 실행 예시

```
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

3. 스태킹 앙상블 실행
main.py를 통해 생성된 OOF 파일들을 사용하여 스태킹 앙상블을 수행합니다. 이 과정은 개별 모델의 성능을 넘어 최종 예측 정확도를 극대화하는 핵심 단계입니다.

# `output/oof/`에 저장된 OOF 파일들을 인자로 전달
```
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

4. 최종 제출 파일 생성
스태킹 앙상블로 생성된 최종 예측 CSV를 대회 제출 형식에 맞게 변환합니다.

```
python output_to_submission.py
```