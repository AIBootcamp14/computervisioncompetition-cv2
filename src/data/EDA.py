#%%
import os
import cv2
import timm
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

data_path = '../../data'

train_img_path = os.path.join(data_path, 'train')
test_img_path = os.path.join(data_path, 'test')

train_img_file_list = [f for f in os.listdir(train_img_path) if f.endswith('.jpg')]
train_img_file_list.sort()

test_img_file_list = [f for f in os.listdir(test_img_path) if f.endswith('.jpg')]
test_img_file_list.sort()

#%%
# train image check (n1 counts from s1)
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
# test image check (n2 counts from s2)
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
# flip
# upside-down
# rotation
# ...

#%%
# Class imbalance check
train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
print(train_df.info())

class_counts = train_df['target'].value_counts().sort_index()

plt.figure(figsize=(10,5))

class_counts.plot(kind='bar')
plt.title("Class Distribution in Train Data")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

print(class_counts)

# minority class: 1, 13, 14

#%%
# Image resolution (Img Size) check
train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
train_longest_sizes = []

for fp in tqdm(train_df['ID']):
    img = cv2.imread(os.path.join(train_img_path,fp))
    h, w = img.shape[:2]
    train_longest_sizes.append(max(h, w))

print("[Train] Max res:", max(train_longest_sizes))
print("[Train] Min res:", min(train_longest_sizes))
print("[Train] Avg res:", sum(train_longest_sizes)/len(train_longest_sizes))

# train:
# Max: 753
# Min: 512
# Avg: 596.31

test = pd.read_csv(os.path.join(data_path, "sample_submission.csv"))
test_longest_sizes = []

for fp in tqdm(test['ID']):
    img = cv2.imread(os.path.join(test_img_path,fp))
    h, w = img.shape[:2]
    test_longest_sizes.append(max(h, w))

print("[Test] Max res:", max(test_longest_sizes))
print("[Test] Min res:", min(test_longest_sizes))
print("[Test] Avg res:", sum(test_longest_sizes)/len(test_longest_sizes))

# test:
# Max: 763
# Min: 512
# Avg: 595.70


#%%
import matplotlib.pyplot as plt

# train height, width
train_heights, train_widths = [], []
for fp in tqdm(train_df['ID']):
    img = cv2.imread(os.path.join(train_img_path, fp))
    h, w = img.shape[:2]
    train_heights.append(h)
    train_widths.append(w)

plt.figure(figsize=(6,6))
plt.scatter(train_widths, train_heights, c="blue", alpha=0.4, s=10)
plt.xlabel("Width (pixels)")
plt.ylabel("Height (pixels)")
plt.title("Train Image Resolution (Width vs Height)")
plt.grid(True)
plt.show()

# test height, width
test_heights, test_widths = [], []
for fp in tqdm(test['ID']):
    img = cv2.imread(os.path.join(test_img_path, fp))
    h, w = img.shape[:2]
    test_heights.append(h)
    test_widths.append(w)

plt.figure(figsize=(6,6))
plt.scatter(test_widths, test_heights, c="red", alpha=0.4, s=10)
plt.xlabel("Width (pixels)")
plt.ylabel("Height (pixels)")
plt.title("Test Image Resolution (Width vs Height)")
plt.grid(True)
plt.show()

#%%
# label noise check (for n*10 counts)
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

# Labels are relatively clean

#%%
# Image check of a class:class_num (count of max_cnt)
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

# %%
# check large image (>=512px) pretrained model in timm & check img size and normalization value
import timm
import pandas as pd
from timm.data import resolve_model_data_config

models_ge512 = []

for name in timm.list_models(pretrained=True):
    try:
        cfg = timm.get_pretrained_cfg(name)
        _, h, w = cfg.input_size
        if (753 >= h >= 512) or (753 >= w >= 512):
            models_ge512.append((name, (h, w)))
    except Exception as e:
        pass

rows = []
for name, size in models_ge512:
    try:
        m = timm.create_model(name, pretrained=True, num_classes=1000, features_only=False)
        data_cfg = resolve_model_data_config(m)
        mean = tuple(map(float, data_cfg["mean"]))
        std  = tuple(map(float, data_cfg["std"]))
        rows.append((name, size, mean, std))
    except Exception as e:
        rows.append((name, size, "ERROR", str(e)))

df = pd.DataFrame(
    [
        {"name": n, "input_size": s, "mean": m, "std": d}
        for n, s, m, d in rows if m != "ERROR"
    ]
).sort_values("name")
print(df)
