import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple
import re
import gc

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    BitsAndBytesConfig
)

# 양자화를 위한 import
try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False
    print("Warning: bitsandbytes not installed. Install with: pip install bitsandbytes")

from data import prepare  # train/val 분리 없는 버전

try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
except Exception:
    sns = None
    plt = None
    classification_report = None
    confusion_matrix = None
    accuracy_score = None


DEFAULT_MODEL_ID = "llava-hf/llava-1.5-7b-hf"


def load_vlm_quantized(model_id: str = DEFAULT_MODEL_ID):
    """4bit 양자화된 VLM 모델 로드 - 메모리 효율적"""

    # 양자화 설정 - 메모리 사용량 대폭 감소
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 4bit 양자화
        bnb_4bit_quant_type="nf4",  # NormalFloat4 양자화
        bnb_4bit_use_double_quant=True,  # 이중 양자화로 추가 압축
        bnb_4bit_compute_dtype=compute_dtype,  # 계산 시 사용할 dtype
    )

    print(f"Loading model with 4-bit quantization...")
    print(f"Compute dtype: {compute_dtype}")

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",  # 자동 디바이스 매핑
        torch_dtype=compute_dtype,
        low_cpu_mem_usage=True,  # CPU 메모리 절약
        max_memory={0: "20GiB"},  # RTX 3090 메모리 제한 (여유분 확보)
    )

    processor = AutoProcessor.from_pretrained(model_id)

    # 모델 정보 출력
    print(f"Model loaded successfully!")
    print(
        f"Model memory footprint: ~{model.get_memory_footprint() / 1e9:.2f} GB")

    return model, processor


def predict_with_text_matching(model, processor, img: Image.Image, id2name: Dict[int, str]) -> Tuple[int, str, Dict]:
    """텍스트 매칭 방식으로 예측 - 메모리 효율적"""

    device = next(model.parameters()).device

    # 이미지 리사이즈 (메모리 절약)
    max_size = 336  # LLaVA의 기본 크기
    img.thumbnail((max_size, max_size), Image.LANCZOS)

    # 전략 1: 문서 설명 요청
    prompt = """Look at this document image and describe what type of document it is. 
Focus on: medical documents, IDs, vehicle documents, receipts, prescriptions, or other official papers.
Be specific about the document type."""

    full_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"

    try:
        inputs = processor(text=full_prompt, images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )

        input_length = inputs['input_ids'].shape[1]
        generated_ids = outputs[0][input_length:]
        description = processor.decode(
            generated_ids, skip_special_tokens=True).lower()

    except Exception as e:
        print(f"Error in generation: {e}")
        description = ""

    # 메모리 정리
    del inputs, outputs
    torch.cuda.empty_cache()

    # 텍스트 매칭으로 클래스 결정
    scores = {}
    for idx, class_name in id2name.items():
        keywords = class_name.lower().replace('_', ' ').split()
        score = 0

        for keyword in keywords:
            if keyword in description:
                score += 1

        # 특별 매칭 규칙
        class_lower = class_name.lower()
        if "id_card" in class_lower and ("identification" in description or "id" in description):
            score += 2
        if "driver" in class_lower and "license" in description:
            score += 3
        if "passport" in class_lower and "passport" in description:
            score += 4
        if "medical" in class_lower and "medical" in description:
            score += 2
        if "vehicle" in class_lower and ("vehicle" in description or "car" in description):
            score += 2
        if "receipt" in class_lower and "receipt" in description:
            score += 2
        if "prescription" in class_lower and "prescription" in description:
            score += 3

        scores[idx] = score

    # 최고 점수 클래스 선택
    best_class = max(scores, key=scores.get)

    # 점수가 모두 0이면 fallback
    if scores[best_class] == 0:
        # 숫자 추출 시도
        numbers = re.findall(r'\b([0-9]|1[0-6])\b', description)
        if numbers:
            try:
                best_class = int(numbers[0])
                if best_class not in id2name:
                    best_class = 0
            except:
                best_class = 0

    return best_class, description, scores


def evaluate_on_train(project_root: Path, model_id: str, sample_size: int = None, batch_size: int = 1):
    """Train 데이터 전체로 평가 (train/val 분리 없음)"""

    root = Path(project_root)
    src_dir = root / "src"
    data_dir = root / "data"
    images_dir = src_dir / "images" / "train"  # 전처리된 train 이미지

    # 메타데이터 읽기
    meta_df, id2name, train_df = prepare(root, force_preprocess=False)

    print(f"Document types: {id2name}")
    print(f"Total training images: {len(train_df)}")

    # 샘플링
    if sample_size and sample_size < len(train_df):
        sample_df = train_df.sample(n=sample_size, random_state=42)
        print(f"Using {len(sample_df)} samples for evaluation")
    else:
        sample_df = train_df
        print("Using all training data for evaluation")

    # 양자화 모델 로드
    if not HAS_BNB:
        print("\n⚠️  WARNING: bitsandbytes not installed. Model will use more memory!")
        print("Install with: pip install bitsandbytes")

    print("\nLoading quantized VLM model...")
    model, processor = load_vlm_quantized(model_id)

    # GPU 메모리 상태 확인
    if torch.cuda.is_available():
        print(
            f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(
            f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    y_true, y_pred = [], []
    descriptions = []

    # 평가 (배치 처리로 메모리 효율화)
    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Evaluating"):
        img_name = row['filename']
        true_label = row['label']

        # 이미지 경로
        img_path = images_dir / img_name
        if not img_path.exists():
            # 원본 이미지 폴백
            img_path = data_dir / "train" / img_name
            if not img_path.exists():
                print(f"Warning: Image {img_name} not found")
                continue

        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                pred_label, desc, scores = predict_with_text_matching(
                    model, processor, img, id2name)

            y_true.append(true_label)
            y_pred.append(pred_label)
            descriptions.append(desc)

            # 처음 10개 상세 출력
            if len(y_true) <= 10:
                print(f"\n[{len(y_true)}] {img_name}:")
                print(f"  True: {true_label} ({id2name[true_label]})")
                print(f"  Pred: {pred_label} ({id2name[pred_label]})")
                print(f"  Description: {desc[:100]}...")
                if true_label == pred_label:
                    print("  ✓ Correct")
                else:
                    print("  ✗ Wrong")

            # 주기적으로 메모리 정리
            if idx % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            continue

    # 결과 분석
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    if accuracy_score:
        acc = accuracy_score(y_true, y_pred)
        print(
            f"Overall Accuracy: {acc:.3f} ({sum([1 for t, p in zip(y_true, y_pred) if t == p])}/{len(y_true)})")

    # 클래스별 성능
    print("\nPer-class performance:")
    for class_id in sorted(id2name.keys()):
        class_indices = [i for i, y in enumerate(y_true) if y == class_id]
        if class_indices:
            class_correct = sum(
                1 for i in class_indices if y_pred[i] == class_id)
            class_acc = class_correct / len(class_indices)
            print(
                f"  {class_id:2d} {id2name[class_id]:40s}: {class_acc:.3f} ({class_correct}/{len(class_indices)})")

    # Classification report
    if classification_report:
        print("\nClassification Report:")
        target_names = [
            f"{i}:{id2name[i][:15]}" for i in sorted(id2name.keys())]
        print(classification_report(y_true, y_pred,
                                    target_names=target_names,
                                    digits=3))

    # Confusion Matrix
    if confusion_matrix and plt and sns:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(17)))

        plt.figure(figsize=(15, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title("Confusion Matrix - Quantized VLM")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        tick_labels = [f"{i}\n{id2name[i][:10]}" for i in range(17)]
        plt.xticks(range(17), tick_labels, rotation=45, ha='right', fontsize=8)
        plt.yticks(range(17), tick_labels, rotation=0, fontsize=8)

        plt.tight_layout()
        matrix_path = src_dir / "confusion_matrix_quantized.png"
        plt.savefig(matrix_path, dpi=150, bbox_inches='tight')
        print(f"\nConfusion matrix saved to: {matrix_path}")
        plt.close()

    # 오분류 분석
    misclassified = [(sample_df.iloc[i]['filename'], y_true[i], y_pred[i], descriptions[i][:100])
                     for i in range(len(y_true)) if y_true[i] != y_pred[i]]

    if misclassified:
        print(f"\n{len(misclassified)} misclassified samples (showing first 5):")
        for fn, true, pred, desc in misclassified[:5]:
            print(f"\n  {fn}:")
            print(f"    True: {true} ({id2name[true]})")
            print(f"    Pred: {pred} ({id2name[pred]})")
            print(f"    Desc: {desc}...")

    # 메모리 상태 최종 확인
    if torch.cuda.is_available():
        print(
            f"\nFinal GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=str, required=True,
                        help="Path to CV_VLM root")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID,
                        help="HuggingFace model ID")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Number of samples to test (None for all)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for processing")
    args = parser.parse_args()

    evaluate_on_train(
        Path(args.project_root),
        args.model_id,
        args.sample_size,
        args.batch_size
    )


if __name__ == "__main__":
    main()
