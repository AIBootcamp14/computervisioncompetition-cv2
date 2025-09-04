import os
import shutil
import pandas as pd

# 파일 경로 설정 (경로가 다르면 수정해 줘)
oof_predictions_path = './model/vit_large_patch16_224/oof_predictions.csv'
train_images_path = './data/train'
output_folder = './misclassified_images'

# 오분류 이미지를 저장할 폴더 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# oof_predictions.csv 파일 읽기
oof_df = pd.read_csv(oof_predictions_path)

# target과 pred가 다른 오분류 데이터만 필터링
misclassified_df = oof_df[oof_df['target'] != oof_df['pred']]

print(f"총 {len(misclassified_df)}개의 오분류 이미지를 찾았습니다.")

# 오분류 이미지를 새 폴더로 복사
for index, row in misclassified_df.iterrows():
    image_id = row['ID']
    source_path = os.path.join(train_images_path, image_id)
    dest_path = os.path.join(output_folder, image_id)
    
    # 파일이 존재하면 복사
    if os.path.exists(source_path):
        shutil.copy(source_path, dest_path)

print(f"오분류된 이미지를 '{output_folder}' 폴더에 모두 복사했습니다.")