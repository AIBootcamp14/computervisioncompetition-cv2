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
s2 = 500
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