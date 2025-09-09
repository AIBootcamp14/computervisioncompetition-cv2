#%%
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

data_path = '../../data'
save_path = os.path.join(data_path, 'noisy')
os.makedirs(save_path, exist_ok=True) 

train_img_path = os.path.join(data_path, 'train')
train_img_file_list = [f for f in os.listdir(train_img_path) if f.endswith('.jpg')]
train_img_file_list.sort()

possible_mislabel_csv = os.path.join(data_path, 'possible_mislabeled_data_mismatch_only.csv')
possible_mislabel_df = pd.read_csv(possible_mislabel_csv)
possible_mislabel_df = possible_mislabel_df[possible_mislabel_df["noisy_flag"] == True][['ID', 'target', 'pred']]

meta_csv = os.path.join(data_path, 'meta.csv')
meta = pd.read_csv(meta_csv)
meta = meta.set_index("target")
meta_dict = meta.to_dict()['class_name']

possible_mislabel_df["target_class_name"] = possible_mislabel_df["target"].map(meta_dict)
possible_mislabel_df["pred_class_name"] = possible_mislabel_df["pred"].map(meta_dict)

for _, row in possible_mislabel_df.iterrows():
    img_path = os.path.join(train_img_path, row["ID"])
    if not os.path.exists(img_path):
        continue

    title = f"labeled {row['target']}: {row['target_class_name']} → predicted {row['pred']}: {row['pred_class_name']}"

    plt.figure(figsize=(8,8))
    img = Image.open(img_path)
    plt.imshow(img)
    plt.title(title, fontsize=14)
    plt.axis("off")
    save_file = os.path.join(save_path, f"{row['ID'].replace('.jpg','')}_noisy.png")
    plt.savefig(save_file, bbox_inches="tight")
    plt.show()

# noisy_flag == True -> Review the data manually and identify each case as either model error or human error -> correct human error labeled data
# 8646f2c3280a4f49.jpg label 7 -> 3
# 1ec14a14bbe633db.jpg label 14 -> 7
# 45f0d2dfc7e47c03.jpg label 3 -> 7
# aec62dced7af97cd.jpg label 3 -> 14
# c5182ab809478f12.jpg label 4 -> 14
