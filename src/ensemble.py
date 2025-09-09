import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

EPS = 1e-9

def _row_normalize(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0)
    s = p.sum(axis=1, keepdims=True)
    s = np.where(s <= 0, 1.0, s)
    return p / s


def _make_features(X, n_models, n_classes, mode):
    feats = []
    for m in range(n_models):
        blk = X[:, m * n_classes:(m + 1) * n_classes]
        blk = _row_normalize(blk)
        if mode == "proba":
            feats.append(blk)
        elif mode == "logproba":
            feats.append(np.log(np.clip(blk, EPS, 1.0)))
        elif mode == "both":
            feats.append(blk)
            feats.append(np.log(np.clip(blk, EPS, 1.0)))
        else:
            raise ValueError("--feature must be one of {proba, logproba, both}")
    return np.concatenate(feats, axis=1)


def _make_standardize(X_oof, X_test, use_std):
    if not use_std:
        return X_oof, X_test
    sc = StandardScaler(with_mean=True, with_std=True)
    X_oof_std = sc.fit_transform(X_oof)
    X_test_std = sc.transform(X_test)
    return X_oof_std, X_test_std


def _load_oof(paths):
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        assert "target" in df.columns, f"OOF must contain 'target': {p}"
        prob_cols = [c for c in df.columns if c.startswith("prob_")]
        assert len(prob_cols) > 0, f"No prob_* columns in OOF: {p}"
        prob_cols = sorted(prob_cols, key=lambda x: int(x.split("_")[1]))
        dfs.append(df[["ID", "target"] + prob_cols].copy())
    return dfs


def _load_test(paths):
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        prob_cols = [c for c in df.columns if c.startswith("prob_")]
        assert len(prob_cols) > 0, f"No prob_* columns in TEST: {p}"
        prob_cols = sorted(prob_cols, key=lambda x: int(x.split("_")[1]))
        dfs.append(df[["ID"] + prob_cols].copy())
    return dfs


def _stack_probs(dfs, kind):
    ids = dfs[0]["ID"]
    for d in dfs[1:]:
        assert (d["ID"].values == ids.values).all(), f"{kind}: ID order must match across files"

    prob_blocks = []
    for d in dfs:
        prob_cols = [c for c in d.columns if c.startswith("prob_")]
        prob_cols = sorted(prob_cols, key=lambda x: int(x.split("_")[1]))
        prob_blocks.append(d[prob_cols].to_numpy(dtype=np.float64))

    X = np.concatenate(prob_blocks, axis=1)
    y = dfs[0]["target"].to_numpy(dtype=np.int64) if "target" in dfs[0].columns else None
    return ids, X, y


def _cv_select_C(X, y, C_grid, n_splits, random_state=42, max_iter=5000, solver="lbfgs", tol=1e-3):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    best_C, best_score = None, -1.0
    for C in C_grid:
        scores = []
        for tr_idx, va_idx in skf.split(X, y):
            clf = LogisticRegression(
                penalty="l2", C=C, max_iter=max_iter,
                multi_class="multinomial", solver=solver, tol=tol
            )
            clf.fit(X[tr_idx], y[tr_idx])
            pred = clf.predict(X[va_idx])
            scores.append(f1_score(y[va_idx], pred, average="macro"))
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score, best_C = mean_score, C
    return best_C, best_score


def _model_key_from_path(path):
    stem = Path(path).stem
    if "." in stem:
        stem = stem.split(".", 1)[0]
    return stem


def _build_output_paths(oof_paths, max_len=40):
    keys, seen = [], set()
    for p in oof_paths:
        k = _model_key_from_path(p)
        if k not in seen:
            keys.append(k); seen.add(k)
    tag = "+".join(keys)
    n = len(keys)

    os.makedirs("output", exist_ok=True)
    if len(tag) <= max_len:
        base = f"ensemble_{tag}"
    else:
        date_str = datetime.now().strftime("%y%m%d-%H%M")
        base = f"{date_str}_ens{n}"

    return os.path.join("output", f"{base}.csv"), os.path.join("output", f"{base}_geomean.csv")


def main():
    ap = argparse.ArgumentParser(description="OOF stacking ensemble (multinomial logistic regression)")
    ap.add_argument("--oof", nargs="+", required=True, help="OOF CSV paths (prob_* & target 포함)")
    ap.add_argument("--test", nargs="+", required=True, help="TEST CSV paths (prob_* 포함)")
    ap.add_argument("--feature", type=str, default="proba", choices=["proba", "logproba", "both"], help="스태킹 특징: proba | logproba | both")
    ap.add_argument("--max_iter", type=int, default=2000, help="LogReg max_iter (기본 2000)")
    ap.add_argument("--standardize", action="store_true", help="OOF/Test 특징 표준화 사용")
    ap.add_argument("--C", type=float, default=None, help="LogReg inverse regularization strength (없으면 CV 탐색)")
    ap.add_argument("--C_grid", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0, 4.0], help="C 탐색 후보값 리스트")
    ap.add_argument("--cv_k", type=int, default=5, help="OOF 내부 CV 폴드 수 (기본 5)")
    ap.add_argument("--target_csv", type=str, default=None, help="sample_submission.csv 경로 (ID 순서 정렬용)")
    ap.add_argument("--also_geomean", action="store_true", help="가중 없는 기하평균 보조 예측 결과물도 같이 생성")
    args = ap.parse_args()

    oof_dfs = _load_oof(args.oof)
    test_dfs = _load_test(args.test)

    oof_ids, X_oof_raw, y_oof = _stack_probs(oof_dfs, "OOF")
    test_ids, X_test_raw, _ = _stack_probs(test_dfs, "TEST")

    n_models = len(oof_dfs)
    n_classes = len([c for c in oof_dfs[0].columns if c.startswith("prob_")])

    X_oof = _make_features(X_oof_raw, n_models, n_classes, args.feature)
    X_test = _make_features(X_test_raw, n_models, n_classes, args.feature)

    X_oof, X_test = _make_standardize(X_oof, X_test, args.standardize)

    if args.C is None:
        C_use, cv_score = _cv_select_C(
            X_oof, y_oof, args.C_grid, n_splits=args.cv_k,
            max_iter=args.max_iter, solver="lbfgs", tol=1e-3
        )
        print(f"[Stacking] CV-selected C = {C_use} (macro F1 = {cv_score:.6f})")
    else:
        C_use = args.C
        print(f"[Stacking] Using given C = {C_use}")

    clf = LogisticRegression(
        penalty="l2",
        C=C_use,
        max_iter=args.max_iter,
        multi_class="multinomial",
        solver="lbfgs",
        tol=1e-3
    )
    clf.fit(X_oof, y_oof)

    prob_test = clf.predict_proba(X_test)
    pred_test = prob_test.argmax(axis=1)

    prob_cols = [f"prob_{i}" for i in range(n_classes)]
    out_df = pd.DataFrame({"ID": test_ids, "target": pred_test})
    out_prob = pd.DataFrame(prob_test, columns=prob_cols)
    out = pd.concat([out_df, out_prob], axis=1)

    if args.target_csv is not None and os.path.exists(args.target_csv):
        ss = pd.read_csv(args.target_csv)
        id_order = ss["ID"].astype(str)
        out["ID"] = out["ID"].astype(str)
        out = out.set_index("ID").loc[id_order].reset_index()

    os.makedirs("output", exist_ok=True)
    out_csv_path, geo_path = _build_output_paths(args.oof)

    out.to_csv(out_csv_path, index=False)
    print(f"[Stacking] Saved: {out_csv_path}")

    coefs = np.abs(clf.coef_)
    block_scores = []

    if args.feature == "both":
        step = 2 * n_classes
        for m in range(n_models):
            s = m * step
            block_p  = coefs[:, s : s + n_classes]
            block_lp = coefs[:, s + n_classes : s + 2 * n_classes]
            block_scores.append((block_p.mean() + block_lp.mean()) / 2.0)
    else:
        for m in range(n_models):
            s = m * n_classes
            block_scores.append(coefs[:, s : s + n_classes].mean())

    norm = np.array(block_scores) / (np.sum(block_scores) + EPS)
    print("[Stacking] Per-model importance (normalized):", norm.tolist())

    if args.also_geomean:
        P_blocks = []
        for m in range(n_models):
            Pm = _row_normalize(X_test_raw[:, m * n_classes:(m + 1) * n_classes])
            P_blocks.append(Pm)
        P = np.stack(P_blocks, axis=0)
        logp = np.log(np.clip(P, EPS, 1.0))
        blended = np.exp(logp.mean(axis=0))
        blended = blended / blended.sum(axis=1, keepdims=True)
        pred_geo = blended.argmax(axis=1)

        out_geo = pd.concat([
            pd.DataFrame({"ID": test_ids, "target": pred_geo}),
            pd.DataFrame(blended, columns=prob_cols)
        ], axis=1)

        if args.target_csv is not None and os.path.exists(args.target_csv):
            ss = pd.read_csv(args.target_csv)
            id_order = ss["ID"].astype(str)
            out_geo["ID"] = out_geo["ID"].astype(str)
            out_geo = out_geo.set_index("ID").loc[id_order].reset_index()

        out_geo.to_csv(geo_path, index=False)
        print(f"[Geomean] Saved: {geo_path}")

if __name__ == "__main__":
    main()
