#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qwen2.5-VL-32B test script (patched with Korean OCR-style document hints)

- 전처리 경로 우선 사용(+옵션으로 PNG 우선)
- Processor use_fast=False (일관성)
- 라벨명 정렬 + 숫자만 출력 강제
- 문서형 이미지는 '이미지에 보이는 한글'만 근거로 라벨 선택 + 클래스별 한국어 키워드 힌트
- 4-bit 양자화(--load_in_4bit)
- --quiet / --dump_suspects / --prefer_png 지원
"""

import os
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter, defaultdict

import torch
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

# HF imports
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
)

# optional import (available in recent transformers)
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
    HAS_QWEN25_NATIVE = True
except Exception:
    HAS_QWEN25_NATIVE = False


# ----------------------------
# Utility: safe int extraction
# ----------------------------
NUM_RE = re.compile(r"\b([0-9]{1,2})\b")


def extract_first_label_int(text: str, num_classes: int) -> int:
    if not isinstance(text, str):
        return -1
    for m in NUM_RE.finditer(text.strip()):
        try:
            v = int(m.group(1))
            if 0 <= v < num_classes:
                return v
        except Exception:
            pass
    return -1


# ----------------------------
# CSV helpers
# ----------------------------
IMG_COL_CANDS = ["image", "img", "file", "filename", "filepath", "path"]
TARGET_COL_CANDS = ["target", "label", "class", "index"]
NAME_COL_CANDS = ["name", "label_name",
                  "class_name", "doc", "type", "category"]


def find_first_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    lc = [c.lower() for c in df.columns]
    for cand in candidates:
        for i, col in enumerate(df.columns):
            if cand in lc[i]:
                return col
    return None


# ----------------------------
# Prompts
# ----------------------------
def build_label_lines(id2name: Dict[int, str]) -> List[str]:
    return [f"{i}: {id2name[i].replace('_', ' ')}" for i in sorted(id2name.keys())]


def build_prompt(id2name: Dict[int, str]) -> str:
    return (
        "다음 이미지를 분류하세요.\n\n"
        "반드시 다음 중 하나의 숫자만 출력하세요 (숫자 외 다른 텍스트 금지):\n"
        + "|".join(str(i) for i in sorted(id2name.keys())) + "\n\n"
        "분류 라벨 목록:\n" + "\n".join(build_label_lines(id2name)) + "\n\n"
        "출력 형식:\n- 오직 하나의 숫자만 출력 (예: 0 또는 12)"
    )


def default_doc_keywords(id2name: Dict[int, str]) -> Dict[int, List[str]]:
    """
    문서형 클래스에 대한 한국어 키워드 힌트.
    - '이미지에 실제로 보이는 한글'만 근거로 판단하도록 유도.
    - 지나치게 포괄적인 단어 대신 현장 데이터에서 자주 보이는 헤더/용어 중심.
    """
    K = {
        0: ["계좌", "계좌번호", "은행", "뱅크", "예금주", "입금계좌", "환불계좌", "ACCOUNT", "BANK"],
        1: ["임신", "출산", "임신부", "산모", "임신출산 진료비", "지원금", "바우처", "국민행복카드", "신청서"],
        3: ["입원확인서", "퇴원확인서", "입·퇴원 확인서", "입원기간", "퇴원일", "병실", "병상"],
        4: ["진단서", "상병명", "상병코드", "진단명"],   # '소견서'는 14로 구분
        6: ["진료비 영수증", "진료비 계산서", "요양급여비용", "진찰료", "처치", "수술료", "진료비"],
        7: ["통원확인서", "외래", "외래진료", "통원", "내원일", "통원치료"],
        8: ["주민등록증", "주민등록", "주민번호", "ID CARD", "NATIONAL ID"],
        9: ["여권", "PASSPORT", "REPUBLIC OF KOREA"],
        10: ["납입 확인서", "납부 확인서", "결제 확인", "입금 확인", "납부영수증", "결제내역"],
        11: ["약국", "약제비", "조제료", "약품명", "복약지도", "약가", "약국 영수증", "영수증(약국)"],
        12: ["처방전", "Rp.", "용법", "용량", "조제", "투약", "일일투여"],
        13: ["이력서", "경력", "학력", "자기소개서"],
        14: ["소견서", "의학적 소견", "의사의 소견"],
        15: ["자동차 등록증", "차량번호", "차대번호", "등록번호"],
        # 2(car_dashboard),16(vehicle_registration_plate),5(driver_lisence)는 이미지형(시각적)이라 제외
    }
    # id2name에 없는 키 제거(혹시 클래스 개수가 다른 경우 대비)
    return {k: v for k, v in K.items() if k in id2name}


def build_doc_prompt(id2name: Dict[int, str]) -> str:
    """
    문서류 전용 프롬프트:
    - 이미지에 보이는 한글 텍스트만 근거
    - 아래 키워드가 보이면 해당 라벨 우선
    - 다중 일치 시 특화 라벨(처방전/진단서/통원확인서 등)이
      포괄 라벨(진료비 영수증/납부확인 등)보다 우선
    - 그래도 불명확하면 가장 일치도가 높은 하나를 선택
    - 6(진료비 영수증), 7(통원확인서)은 최후의 선택지
    """
    kw = default_doc_keywords(id2name)

    lines = []
    for i in sorted(id2name.keys()):
        name = id2name[i].replace("_", " ")
        if i in kw:
            hint = " | ".join(kw[i])
            lines.append(f"- {i}: {name}  ← 키워드 예: {hint}")
        else:
            lines.append(f"- {i}: {name}")

    rule = (
        "지침:\n"
        "1) 이미지에서 실제로 읽히는 한글(또는 영어)이 위 키워드와 일치/유사하면 해당 라벨을 선택합니다.\n"
        "2) 둘 이상 일치하면 특화 문서(예: '처방전','진단서','통원확인서','여권','주민등록증')를 "
        "포괄 문서(예: '진료비 영수증','납부 확인')보다 우선합니다.\n"
        "3) 약 관련 단어(약국/조제/약품명/복약지도)가 보이면 '의약품 영수증(11)' 또는 '처방전(12)'을 우선 검토합니다.\n"
        "4) 제목/도장/헤더 영역 텍스트를 먼저 보고, 표·내역은 보조로 사용합니다.\n"
        "5) 판단 근거는 이미지 내 텍스트만 사용하고, 외부지식/추측은 금지합니다.\n"
    )

    return (
        "아래 문서 이미지를 분류하세요.\n"
        "반드시 아래 라벨 번호 중 하나만 숫자로 출력하세요 (숫자 외 다른 텍스트 금지).\n\n"
        "라벨 목록 및 키워드 힌트:\n" + "\n".join(lines) + "\n\n" +
        rule +
        "출력 형식: 위 목록의 번호 하나만 숫자로 출력 (예: 0 또는 12)"
    )


# ----------------------------
# Loader for Qwen2.5-VL
# ----------------------------
def load_qwen2_vl(
    model_id: str,
    load_in_4bit: bool = True,
    device_map: str = "auto",
    offload_dir: str = "offload_cache",
    flash_attn: bool = False,
    gpu_mem_gib: int = 22,   # GPU 사용 한도(3090 24GB → 22~23GiB 권장)
    cpu_mem_gib: int = 64,   # CPU 오프로딩 한도
):
    import os
    import torch
    from transformers import (
        AutoProcessor, BitsAndBytesConfig,
        AutoModelForCausalLM
    )
    # 새 오토클래스(있으면 우선), 없으면 Vision2Seq 폴백
    try:
        from transformers import AutoModelForImageTextToText as AutoModelForVLM
        HAS_IMGTXT = True
    except Exception:
        HAS_IMGTXT = False
        from transformers import AutoModelForVision2Seq as AutoModelForVLM

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        HAS_QWEN25_NATIVE = True
    except Exception:
        HAS_QWEN25_NATIVE = False

    print("Loading Qwen2.5-VL model (32B target)...")

    # ---- 4-bit 양자화 설정 ----
    bnb_config = None
    if load_in_4bit:
        print("Loading with 4-bit quantization (bitsandbytes, compute fp16)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,  # Ampere(3090)에서 안정적
            bnb_4bit_use_double_quant=True,
        )
    else:
        print("Loading in full/half precision (VRAM required)...")

    # ---- Processor ----
    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True, use_fast=False
    )

    # ---- max_memory: GPU 키는 'cuda:0'가 아니라 정수 0,1,... 로! ----
    n_gpu = torch.cuda.device_count()
    max_memory = {i: f"{gpu_mem_gib}GiB" for i in range(
        n_gpu)} if n_gpu > 0 else {}
    max_memory["cpu"] = f"{cpu_mem_gib}GiB"
    os.makedirs(offload_dir, exist_ok=True)

    common_kwargs = dict(
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=bnb_config,
        max_memory=max_memory,          # ← 정수 키 사용
        offload_folder=offload_dir,
        low_cpu_mem_usage=True,
    )

    model = None
    tried = []

    # 1) Qwen2.5-VL 전용 클래스
    if ("Qwen2.5-VL" in model_id or "Qwen2_5" in model_id) and HAS_QWEN25_NATIVE:
        tried.append("Qwen2_5_VLForConditionalGeneration")
        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id, **common_kwargs
            )
        except Exception as e:
            print(f"[warn] Qwen2_5_VLForConditionalGeneration failed: {e}")

    # 2) AutoModelForCausalLM
    if model is None:
        tried.append("AutoModelForCausalLM")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, **common_kwargs)
        except Exception as e:
            print(f"[warn] AutoModelForCausalLM failed: {e}")

    # 3) 비전-텍스트 폴백 (ImageTextToText가 있으면 우선)
    if model is None:
        tried.append(
            "AutoModelForImageTextToText" if HAS_IMGTXT else "AutoModelForVision2Seq")
        model = AutoModelForVLM.from_pretrained(model_id, **common_kwargs)

    print(f"Model loaded via: {tried[-1]}")

    # pad_token 보정
    if getattr(model.config, "pad_token_id", None) is None and hasattr(processor, "tokenizer"):
        model.config.pad_token_id = processor.tokenizer.eos_token_id

    # (선택) Flash-Attn2
    if flash_attn:
        try:
            model.config.attn_implementation = "flash_attention_2"
            print("Flash-Attention 2 enabled.")
        except Exception as e:
            print(f"[warn] flash_attention_2 not enabled: {e}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        alloc = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"Model memory: ~{alloc:.2f} GB | Reserved: {reserved:.2f} GB")

    return model, processor


# ----------------------------
# DOC / SCENE quick check
# ----------------------------
def is_document_like(im, model, processor) -> bool:
    messages = [
        {"role": "system", "content": "Answer with exactly one word: DOC or SCENE."},
        {"role": "user", "content": [
            {"type": "text", "text": "이 이미지는 인쇄된 글자가 주된 문서입니까? DOC 또는 SCENE 한 단어만 출력하세요."},
            {"type": "image"},
        ]}
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[im], return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=2,
            do_sample=False,
            eos_token_id=getattr(processor, "tokenizer", None).eos_token_id if hasattr(
                processor, "tokenizer") else None,
            pad_token_id=model.config.pad_token_id,
        )
    ans = processor.batch_decode(
        out[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)[0].strip().upper()
    return "DOC" in ans


# ----------------------------
# Single prediction
# ----------------------------
def predict_one_image(
    img_path: Path,
    model,
    processor,
    prompt_text: str,
    num_classes: int,
    id2name: dict[int, str],
    gen_max_new_tokens: int = 3,
) -> Tuple[int, str]:
    with Image.open(img_path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")

    # 문서 여부 → 프롬프트 선택 (문서면 한국어 키워드 힌트 포함)
    doclike = is_document_like(im, model, processor)
    user_text = build_doc_prompt(id2name) if doclike else prompt_text

    messages = [
        {"role": "system", "content": "You are a precise visual classification model. Respond with one number only."},
        {"role": "user", "content": [
            {"type": "text", "text": user_text},
            {"type": "image"},
        ]},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[im], return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=gen_max_new_tokens,
            do_sample=False,
            eos_token_id=getattr(processor, "tokenizer", None).eos_token_id if hasattr(
                processor, "tokenizer") else None,
            pad_token_id=model.config.pad_token_id,
        )
    gen_text = processor.batch_decode(
        output_ids[:, inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )[0].strip()

    pred = extract_first_label_int(gen_text, num_classes)
    return pred, gen_text


# ----------------------------
# Evaluate over train.csv
# ----------------------------
def evaluate_on_train(
    project_root: Path,
    train_csv: Path,
    processed_train_dir: Path,
    raw_train_dir: Path,
    meta_csv: Path = None,
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    load_in_4bit: bool = True,
    sample_size: int = 0,
    quiet: bool = False,
    dump_suspects: str = "",
    prefer_png: bool = False,
):
    # --- load CSVs
    train_df = pd.read_csv(train_csv)
    img_col = find_first_col(train_df, IMG_COL_CANDS)
    if img_col is None:
        for col in train_df.columns:
            if str(col).lower() in ("id", "image_id", "img_id"):
                img_col = col
                break
    tgt_col = find_first_col(train_df, TARGET_COL_CANDS)
    if img_col is None or tgt_col is None:
        raise ValueError("train.csv에서 이미지 칼럼 또는 라벨 칼럼을 찾지 못했습니다.")

    # id2name
    id2name = {}
    if meta_csv is not None and meta_csv.exists():
        meta_df = pd.read_csv(meta_csv)
        name_col = find_first_col(meta_df, NAME_COL_CANDS)
        meta_tgt_col = find_first_col(meta_df, TARGET_COL_CANDS)
        if name_col is not None and meta_tgt_col is not None:
            for _, r in meta_df.iterrows():
                try:
                    id2name[int(r[meta_tgt_col])] = str(r[name_col])
                except Exception:
                    pass
    if not id2name:
        classes = sorted(set(int(v) for v in train_df[tgt_col].tolist()))
        id2name = {i: f"class_{i}" for i in classes}

    num_classes = len(id2name)
    print(f"Total images in train.csv: {len(train_df)}")
    print(f"Number of classes: {num_classes}")
    print("Document types:", {k: v for k, v in id2name.items()})

    # sample
    if sample_size and sample_size > 0:
        train_df = train_df.sample(
            n=sample_size, random_state=42).reset_index(drop=True)
        print(f"Using a subset: {len(train_df)}")
    else:
        print("Using all training data")

    # load model
    model, processor = load_qwen2_vl(
        model_id,
        load_in_4bit=load_in_4bit,
        device_map="auto",
        offload_dir="offload_cache",
        flash_attn=False,          # fa2 설치했으면 True
    )

    generic_prompt = build_prompt(id2name)

    # eval loop
    y_true, y_pred = [], []
    wrong_samples = []
    pred_counter = Counter()
    per_class_confuse = defaultdict(Counter)

    progress = tqdm(range(len(train_df)), total=len(
        train_df), desc="Evaluating", ncols=100)
    for idx in progress:
        row = train_df.iloc[idx]
        rel_path = str(row[img_col])

        # Path selection: preprocessed first (PNG 우선 옵션)
        rel = Path(rel_path)
        img_path = None
        if prefer_png:
            p_png = processed_train_dir / rel.with_suffix(".png")
            if p_png.exists():
                img_path = p_png
        if img_path is None:
            p_proc = processed_train_dir / rel
            if p_proc.exists():
                img_path = p_proc
        if img_path is None:
            p_raw = raw_train_dir / rel
            if p_raw.exists():
                img_path = p_raw
        if img_path is None:
            p_abs = Path(rel_path)
            if p_abs.exists():
                img_path = p_abs
        if img_path is None:
            if not quiet:
                print(f"[MISS] {rel_path}")
            continue

        true_id = int(row[tgt_col])

        pred_id, raw_out = predict_one_image(
            img_path=img_path,
            model=model,
            processor=processor,
            prompt_text=generic_prompt,
            num_classes=num_classes,
            id2name=id2name,
            gen_max_new_tokens=3,
        )

        y_true.append(int(true_id))
        y_pred.append(pred_id)
        pred_counter[pred_id] += 1
        per_class_confuse[true_id][pred_id] += 1

        if not quiet:
            base = os.path.basename(rel_path)
            true_name = id2name.get(true_id, str(true_id))
            pred_name = id2name.get(pred_id, str(pred_id))
            check = "✓ Correct" if pred_id == true_id else "✗ Wrong"
            print(
                f"\n[{idx+1}] {base}:\n"
                f"  True: {true_id} ({true_name})\n"
                f"  Pred: {pred_id} ({pred_name})\n"
                f"  Raw: '{raw_out}'\n"
                f"  {check}"
            )

        if pred_id != true_id and len(wrong_samples) < 2000:
            wrong_samples.append((rel_path, true_id, pred_id, raw_out))

    # metrics
    acc = accuracy_score(y_true, y_pred)
    correct_cnt = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS - Qwen2.5-VL (Patched)")
    print("=" * 60)
    print(f"Overall Accuracy: {acc:.3f} ({correct_cnt}/{len(y_true)})\n")

    pred_dist_str = ", ".join(
        [f"{k}: {v}" for k, v in sorted(pred_counter.items())])
    print(f"Prediction distribution: {{{pred_dist_str}}}\n")

    print("Per-class performance:")
    supports = Counter(y_true)
    corrects = Counter([c for c, p in zip(y_true, y_pred) if c == p])
    for cid in sorted(id2name.keys()):
        sup = supports[cid]
        cor = corrects[cid]
        recall = (cor / sup) if sup > 0 else 0.0
        print(f"{cid:>4} {id2name[cid]:<40}: {recall:0.3f} ({cor}/{sup})")
        confuse_sorted = sorted(
            per_class_confuse[cid].items(), key=lambda x: x[1], reverse=True)
        if sup > 0:
            tops = [(k, v) for k, v in confuse_sorted if k != cid][:3]
            if tops:
                print(f"     → Most predicted as: {tops}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=sorted(id2name.keys()),
            target_names=[f"{i}:{id2name[i]}" for i in sorted(id2name.keys())],
            digits=3,
            zero_division=0,
        )
    )

    if dump_suspects:
        import csv
        out_path = Path(dump_suspects)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["filename", "true_id", "true_name",
                       "pred_id", "pred_name", "raw_output"])
            for rel_path, t, p, raw in wrong_samples:
                w.writerow([rel_path, t, id2name.get(
                    t, t), p, id2name.get(p, p), raw])
        print(f"\nSaved suspect samples to {out_path}")

    return acc


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, default=".")
    ap.add_argument("--train_csv", type=str, default="data/train.csv")
    ap.add_argument("--meta_csv", type=str, default="data/meta.csv")
    ap.add_argument("--processed_train_dir", type=str,
                    default="src/images/train")
    ap.add_argument("--raw_train_dir", type=str, default="data/train")
    ap.add_argument("--model_id", type=str,
                    default="Qwen/Qwen2.5-VL-32B-Instruct")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--sample_size", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--dump_suspects", type=str, default="")
    ap.add_argument("--prefer_png", action="store_true",
                    help="전처리 경로에 PNG가 있으면 우선 사용")
    args = ap.parse_args()

    project_root = Path(args.project_root)
    train_csv = (project_root / args.train_csv).resolve()
    meta_csv = (project_root / args.meta_csv).resolve()
    processed_train_dir = (project_root / args.processed_train_dir).resolve()
    raw_train_dir = (project_root / args.raw_train_dir).resolve()

    print(f"Total images in {args.train_csv}: {len(pd.read_csv(train_csv))}")
    if processed_train_dir.exists():
        print("Preprocessed images directory found. Using it with highest priority.")
    else:
        print("No preprocessed directory found; will fall back to raw images.")

    evaluate_on_train(
        project_root=project_root,
        train_csv=train_csv,
        processed_train_dir=processed_train_dir,
        raw_train_dir=raw_train_dir,
        meta_csv=meta_csv if meta_csv.exists() else None,
        model_id=args.model_id,
        load_in_4bit=args.load_in_4bit,
        sample_size=args.sample_size,
        quiet=args.quiet,
        dump_suspects=args.dump_suspects,
        prefer_png=args.prefer_png,
    )


if __name__ == "__main__":
    main()
