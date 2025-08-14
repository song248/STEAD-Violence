import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

GT_DIR   = "hf-violence/GT" 
PRED_DIR = "hf-violence/post-predict"

def _read_csv(path, expected_cols):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    keep = [c for c in expected_cols if c in df.columns]
    if not keep:
        if "frame" not in df.columns:
            raise ValueError(f"Missing 'frame' column: {path}")
        for c in expected_cols:
            if c not in df.columns:
                df[c] = np.nan
        keep = expected_cols
    df = df[keep]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["frame"])
    df["frame"] = df["frame"].astype(np.int64)
    df = df.sort_values("frame").groupby("frame", as_index=False).last()
    return df

def _load_gt(path):
    df = _read_csv(path, ["frame", "violence"])
    if "violence" not in df.columns:
        raise ValueError(f"GT file missing 'violence' column: {path}")
    df["violence"] = (df["violence"] > 0.5).astype(np.int64)
    return df[["frame", "violence"]]

def _load_pred(path, default_thresh=0.5):
    df = _read_csv(path, ["frame", "violence", "prob"])
    if "violence" in df.columns and df["violence"].notna().any():
        v = df["violence"].fillna(0).to_numpy()
        v = (v > 0.5).astype(np.int64)
        df["violence"] = v
    elif "prob" in df.columns:
        p = df["prob"].astype(float).to_numpy()
        p = np.clip(p, 0.0, 1.0)
        df["violence"] = (p >= default_thresh).astype(np.int64)
    else:
        raise ValueError(f"Prediction file has neither 'violence' nor 'prob': {path}")
    return df[["frame", "violence"]]

def evaluate_micro(gt_dir, pred_dir):
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)

    gt_files = {Path(p).name: p for p in glob.glob(str(gt_dir / "*.csv"))}
    pr_files = {Path(p).name: p for p in glob.glob(str(pred_dir / "*.csv"))}

    common = sorted(set(gt_files.keys()) & set(pr_files.keys()))
    if not common:
        raise SystemExit(f"No matching filenames between {gt_dir} and {pred_dir}")

    TP = FP = TN = FN = 0
    total_frames = 0
    used_files = 0
    skipped = []

    for fname in tqdm(common, desc="Evaluating", unit="file"):
        try:
            df_gt = _load_gt(gt_files[fname])
            df_pr = _load_pred(pr_files[fname])
        except Exception as e:
            skipped.append((fname, str(e)))
            continue

        df = pd.merge(df_gt, df_pr, on="frame", how="left", suffixes=("_gt", "_pred"))
        df = df.sort_values("frame").reset_index(drop=True)

        if "violence_pred" not in df.columns:
            df["violence_pred"] = np.nan
        df["violence_pred"] = df["violence_pred"].fillna(0).astype(np.int64)
        df["violence_gt"] = df["violence_gt"].astype(np.int64)

        y = df["violence_gt"].to_numpy()
        yhat = df["violence_pred"].to_numpy()

        tp = int(np.sum((y == 1) & (yhat == 1)))
        tn = int(np.sum((y == 0) & (yhat == 0)))
        fp = int(np.sum((y == 0) & (yhat == 1)))
        fn = int(np.sum((y == 1) & (yhat == 0)))

        TP += tp; TN += tn; FP += fp; FN += fn
        total_frames += len(df)
        used_files += 1

    if used_files == 0:
        raise SystemExit("No files evaluated successfully.")

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_micro  = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0
    accuracy  = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0

    print("\n=== Confusion Matrix (micro, aggregated over all frames) ===")
    print(f"            Pred 0     Pred 1")
    print(f"True 0   :  {TN:8d}  {FP:8d}")
    print(f"True 1   :  {FN:8d}  {TP:8d}")

    print("\n=== Metrics (micro) ===")
    print(f"Files used     : {used_files}")
    if skipped:
        print(f"Files skipped  : {len(skipped)}")
        for fname, err in skipped:
            print(f"  - {fname}: {err}")
    print(f"Total frames   : {total_frames}")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1-score(micro): {f1_micro:.4f}")
    print(f"Accuracy       : {accuracy:.4f}")

if __name__ == "__main__":
    evaluate_micro(GT_DIR, PRED_DIR)
