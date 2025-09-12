import argparse
import numpy as np
import pandas as pd

EPS = 1e-9

def load_oof(path):
    df = pd.read_csv(path)
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    prob_cols = sorted(prob_cols, key=lambda x: int(x.split("_")[1]))
    P = df[prob_cols].to_numpy(dtype=np.float64)
    P = np.clip(P, EPS, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    return df[["ID","target"]].copy(), P, prob_cols

def geomean(stack):
    logp = np.log(np.clip(stack, EPS, 1.0))
    out = np.exp(logp.mean(axis=0))
    out /= out.sum(axis=1, keepdims=True)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", nargs="+", required=True)
    ap.add_argument("--out_csv", default="noisy_candidates.csv")
    ap.add_argument("--topk_per_class", type=int, default=20)
    ap.add_argument("--agree_min", type=int, default=2, help="여러 OOF일 때 동일 예측 동의 최소 모델수")
    ap.add_argument("--mismatch_only", action="store_true", help="target != pred 인 샘플만 대상으로 후보 추출")
    args = ap.parse_args()

    metas, probs, prob_cols = [], [], None
    for p in args.oof:
        meta, P, prob_cols = load_oof(p)
        metas.append(meta)
        probs.append(P)

    base = metas[0]
    for m in metas[1:]:
        assert (m["ID"].values == base["ID"].values).all()
        assert (m["target"].values == base["target"].values).all()

    ids = base["ID"].astype(str).values
    y = base["target"].to_numpy()

    M, N, C = len(probs), probs[0].shape[0], probs[0].shape[1]
    stack = np.stack(probs, axis=0) 

    preds = stack.argmax(axis=2)

    vote_max = []
    vote_cls = []
    for i in range(N):
        v = np.bincount(preds[:, i], minlength=C)
        vote_max.append(v.max())
        vote_cls.append(np.argmax(v)) 

    votes = np.array(vote_max) 
    vote_cls = np.array(vote_cls) 

    Pgm = geomean(stack)
    p_true = Pgm[np.arange(N), y]
    pred = Pgm.argmax(axis=1)
    p_pred = Pgm[np.arange(N), pred]
    sort_idx = np.argsort(-Pgm, axis=1)
    top1 = Pgm[np.arange(N), sort_idx[:,0]]
    top2 = Pgm[np.arange(N), sort_idx[:,1]]
    margin = top1 - top2
    entropy = -(Pgm*np.log(Pgm+EPS)).sum(axis=1)
    ce_loss = -np.log(p_true + EPS)

    df = pd.DataFrame({
        "ID": ids,
        "target": y,
        "pred": pred,
        "agree": votes,
        "agree_class": vote_cls,
        "agree_ratio": votes / float(M),
        "p_true": p_true,
        "p_pred": p_pred,
        "margin": margin,
        "entropy": entropy,
        "ce_loss": ce_loss
    })

    ruleA = (df["pred"] != df["target"]) & (df["p_pred"] >= 0.9) & (df["margin"] >= 0.2)
    ruleB = (df["p_true"] <= 0.1) & (df["ce_loss"] >= 2.3)
    ruleC = (df["agree"] >= args.agree_min) if M >= args.agree_min else True

    df["noisy_flag"] = (ruleA | ruleB) & ruleC

    base_df = df
    if args.mismatch_only:
        base_df = base_df[base_df["pred"] != base_df["target"]]

    out_rows = []
    for c in sorted(base_df["target"].unique()):
        sub = base_df[base_df["target"] == c].sort_values(["noisy_flag","ce_loss"], ascending=[False, False])
        out_rows.append(sub.head(args.topk_per_class))
    out = pd.concat(out_rows)
    out.to_csv(args.out_csv, index=False)
    print(f"[noisy] Saved: {args.out_csv} (M={M}, N={N}, flagged={int(out['noisy_flag'].sum())})")

if __name__ == "__main__":
    main()