#%%
import os
import time

import cv2
import timm
import torch
import albumentations as A
import pandas as pd
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from albumentations.pytorch import ToTensorV2
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

#%%
# EDA
data_path = '../../data'

train_img_path = os.path.join(data_path, 'train')
test_img_path = os.path.join(data_path, 'test')

train_img_file_list = [f for f in os.listdir(train_img_path) if f.endswith('.jpg')]
train_img_file_list.sort()

test_img_file_list = [f for f in os.listdir(test_img_path) if f.endswith('.jpg')]
test_img_file_list.sort()

#%%
# train 이미지 살펴보기 (s1 부터 n1개)
s1 = 0
n1 = 100

plt.figure(figsize=(30,150))

for i, f in enumerate(train_img_file_list[s1:s1 + n1]):
  plt.subplot(n1//4 + 1, 4,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  img = Image.open(os.path.join(train_img_path,f))
  plt.imshow(img)
plt.show()

#%%
# test 이미지 살펴보기 (s2 부터 n2개)
s2 = 1400
n2 = 100

plt.figure(figsize=(30,150))

for i, f in enumerate(test_img_file_list[s2: s2+n2]):
  plt.subplot(n2//4 + 1, 4,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  img = Image.open(os.path.join(test_img_path,f))
  plt.imshow(img)
plt.show()

# noise
# 좌우반전
# 상하반전
# 회전
# 위에 것 여러 개 결합

#%%
# Class 불균형 확인
train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
print(train_df.info())

class_counts = train_df['target'].value_counts().sort_index()

plt.figure(figsize=(10,5))

class_counts.plot(kind='bar')
plt.title("Class Distribution in Train Data")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# 클래스별 개수 확인
print(class_counts)

# class 1, 13, 14 에 불균형 존재 확인

#%%
# 문서 중 가장 큰 사이즈 확인
longest_sizes = []

for fp in tqdm(train_df['ID']):
    img = cv2.imread(os.path.join(train_img_path,fp))
    h, w = img.shape[:2]
    longest_sizes.append(max(h, w))

print("최대 이미지 크기:", max(longest_sizes))
print("최소 이미지 크기:", min(longest_sizes))
print("평균 이미지 크기:", sum(longest_sizes)/len(longest_sizes))

# 최대 이미지 크기: 753
# 최소 이미지 크기: 512
# 평균 이미지 크기: 596.31

#%%
# label noise 확인
meta_path = os.path.join(data_path, 'meta.csv')
meta = pd.read_csv(meta_path)
meta = meta.set_index("target")
meta_dict = meta.to_dict()['class_name']

train_df_copy = train_df.copy()
train_df_copy["class_name"] = train_df_copy["target"].map(meta_dict)
train_df_copy

# s3 = 0
n = 10

for s3 in range(0, n):
    plt.figure(figsize=(40, 80))   

    for i, f in enumerate(train_img_file_list[s3*10:s3*10+10]):
        class_name = train_df_copy[train_df_copy["ID"]== f]["class_name"].reset_index(drop=True).get(0)
        plt.subplot(5, 2, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        img = Image.open(os.path.join(train_img_path,f))
        plt.title(class_name, fontsize=80)
        plt.imshow(img)
        plt.tight_layout()
    plt.show()

# label noise는 심하지 않음 (구분하기 애매한 사진은 없음)

#%%
# 특정 class의 사진들만 max_cnt장 보기
class_num = 14
max_cnt = 5


cnt = 0
for i, r in train_df.iterrows():
    df_id, df_class_name = r
    if df_class_name == class_num:
        cnt += 1
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        img = Image.open(os.path.join(train_img_path,df_id))
        plt.title(f"{df_id}_{df_class_name}", fontsize=30)
        plt.imshow(img)
        plt.tight_layout()
        plt.show()
    if cnt >= max_cnt:
       break

# 문제점: 특히 7번 class를 잘 구별하지 못함 -> 왜?
# 같은 의료문서인 3, 4, 6, 7, 11, 12인 애들은 구별 잘 함
# 확인 결과 다른 의료문서들은 헤더가 일정하거나 포맷이 일정
# 하지만 7번 확인서의 경우는 헤더부터 다양함
# 7번보다는 훨씬 덜하지만 항상 예측 잘못한 것에 끼는 14번도 살펴보면 7번과 굉장히 비슷한 형태
## 시도해 볼 해결책 1) 이미 돌린 모델들 중 이걸 잘 잡아내서 7번도 잘 예측한 모델을 앙상블 (근본적인 해결책은 x)
## 시도해 볼 해결책 2) 해상도 높이기
## 시도해 볼 해결책 3) 미세한 글씨체를 잡아내는데 좋은 모델을 써보기
## 시도해 볼 해결책 4) 문서형태가 7,14번일 때는 의도적인 훼손을 줄이기(특히 헤더를 의도적으로 많이 살려보기)

# %%
