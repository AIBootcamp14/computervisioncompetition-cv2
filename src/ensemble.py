import os
import pandas as pd
import numpy as np

######################################################

# 앙상블할 모델과 가중치
models = {
    "coatnet_1_rw_224": 0.2,
    "convnextv2_base": 0.5,
    "tf_efficientnet_b4_ns": 0.3,
    # "convnext_base": 0.0,
    # "swinv2_base_window12to24_192to384": 0.0,
}

#######################################################

model_names = [m for m, w in models.items() if w > 0]
if not model_names:
    raise ValueError("No valid model with weight > 0")

ss_path = "/root/project/cv_competition/data/sample_submission.csv" 
paths = [f"output/{m}.csv" for m in model_names]
out_path = f"output/ensemble_({'+'.join(model_names)}).csv"

# sample submission과 ID 순서 통일
ss = pd.read_csv(ss_path)
id_order = ss["ID"].astype(str)

## 모델 결과 불러오기
dfs = [pd.read_csv(p) for p in paths]
for i in range(len(dfs)):
    dfs[i]["ID"] = dfs[i]["ID"].astype(str)

## ID 동일여부 확인
id_set = set(id_order)
for i, df in enumerate(dfs):
    if set(df["ID"]) != id_set:
        raise ValueError(f"{paths[i]}: ID mismatch with sample_submission.csv")

# 확률 컬럼 가져오기
proba_cols = [c for c in dfs[0].columns if c.startswith("prob_")]
if not proba_cols:
    raise ValueError("prob_* columns not found.")

# 가중치 정규화
w = np.array([models[m] for m in model_names], dtype=float)
if w.sum() <= 0:
    raise ValueError("Sum of weight <= 0.")
w = w / w.sum()

# ID를 문자열로 통일하고 인덱스 설정
for i in range(len(dfs)):
    dfs[i]["ID"] = dfs[i]["ID"].astype(str)
    dfs[i] = dfs[i].set_index("ID")

# 샘플 제출의 ID가 모든 예측에 존재하는지 체크
missing_by_model = []
for i in range(len(dfs)):
    missing = id_order[~id_order.isin(dfs[i].index)]
    if len(missing) > 0:
        missing_by_model.append((paths[i], len(missing)))
if missing_by_model:
    msg = ", ".join([f"{p}: {n} missing IDs" for p, n in missing_by_model])
    raise ValueError(f"Some IDs in sample_submission are missing in model outputs: {msg}")

# 샘플 제출 순서에 맞춰 확률 행렬 추출
mats = [df.loc[id_order.tolist(), proba_cols].to_numpy() for df in dfs]
blend = sum(w[i] * mats[i] for i in range(len(dfs)))

# 최종 제출 파일 생성
final = pd.DataFrame({
    "ID": id_order,
    "target": blend.argmax(axis=1).astype(int)
})
final.to_csv(out_path, index=False)
print(f"Saved: {out_path} (order matches {ss_path})")

# 아이디 순서 일치 여부 확인
sample_ids = pd.read_csv(ss_path)["ID"].tolist()
ensemble_ids = pd.read_csv(out_path)["ID"].tolist()

if sample_ids == ensemble_ids:
    print("일치")
else:
    print("불일치")
    for i, (s_id, e_id) in enumerate(zip(sample_ids, ensemble_ids)):
        if s_id != e_id:
            print(f"위치 {i} 불일치: sample={s_id}, ensemble={e_id}")
    if len(sample_ids) != len(ensemble_ids):
        print(f"길이 불일치. sample={len(sample_ids)}, ensemble={len(ensemble_ids)}")