import os
import glob
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def load_csv_as_df(path, cols_expected):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df[[c for c in cols_expected]]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["frame"])
    df["frame"] = df["frame"].astype(np.int64)
    df = df.sort_values("frame").groupby("frame", as_index=False).last()
    return df

def best_f1_threshold(y_true: np.ndarray, p: np.ndarray):
    p = np.clip(p.astype(np.float64), 0.0, 1.0)
    y = (y_true > 0.5).astype(np.int64)

    uniq = np.unique(p)
    candidates = np.concatenate(([-np.inf], uniq, [np.inf]))

    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        yhat = (p >= t).astype(np.int64)
        tp = np.sum((yhat == 1) & (y == 1))
        fp = np.sum((yhat == 1) & (y == 0))
        fn = np.sum((yhat == 0) & (y == 1))
        denom = 2 * tp + fp + fn
        f1 = 0.0 if denom == 0 else (2.0 * tp) / denom
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, float(best_f1)

def main():
    ap = argparse.ArgumentParser(description="Per-file F1-optimal thresholding with GT-length alignment")
    ap.add_argument("--gt_dir", type=str, default="hf-violence/GT",
                    help="GT csv")
    ap.add_argument("--prob_dir", type=str, default="hf-violence/predict_hf_1sec_10fps",
                    help="prob csv")
    ap.add_argument("--out_pred_dir", type=str, default="hf-violence/predict",
                    help="output save dir")
    ap.add_argument("--report_name", type=str, default="report.csv",
                    help="")
    args = ap.parse_args()

    gt_dir = Path(args.gt_dir)
    prob_dir = Path(args.prob_dir)
    out_dir = Path(args.out_pred_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_files = {Path(p).name for p in glob.glob(str(gt_dir / "*.csv"))}
    prob_files = {Path(p).name for p in glob.glob(str(prob_dir / "*.csv"))}
    common = sorted(gt_files & prob_files)
    if len(common) == 0:
        raise SystemExit(f"No matching csv filenames between {gt_dir} and {prob_dir}")

    report_rows = []

    for fname in tqdm(common, desc="Per-file thresholding", unit="file"):
        gt_path = gt_dir / fname
        pr_path = prob_dir / fname

        df_gt = load_csv_as_df(gt_path, cols_expected=["frame", "violence"])
        df_pr = load_csv_as_df(pr_path, cols_expected=["frame", "prob"])

        df = pd.merge(df_gt, df_pr, on="frame", how="left")
        df = df.sort_values("frame").reset_index(drop=True)

        if "prob" not in df.columns:
            df["prob"] = np.nan
        df["prob"] = df["prob"].astype(float)
        df["prob"] = df["prob"].ffill().bfill()
        df["prob"] = df["prob"].fillna(0.0)
        df["prob"] = df["prob"].clip(0.0, 1.0)

        y_true = (df["violence"].to_numpy() > 0.5).astype(np.int64)
        p = df["prob"].to_numpy().astype(np.float64)
        best_t, best_f1 = best_f1_threshold(y_true, p)

        y_pred = (p >= best_t).astype(np.int64)
        out_df = pd.DataFrame({"frame": df["frame"].astype(np.int64), "violence": y_pred})
        out_df.to_csv(out_dir / fname, index=False)

        report_rows.append({
            "filename": fname,
            "best_threshold": best_t,
            "best_f1": best_f1,
            "num_frames": int(len(df)),
            "num_pos_gt": int(y_true.sum())
        })

    report_path = prob_dir / args.report_name
    rep_df = pd.DataFrame(report_rows).sort_values("filename")
    rep_df.to_csv(report_path, index=False)

    print(f"[DONE] Saved predictions to: {out_dir.resolve()}")
    print(f"[DONE] Saved report to    : {report_path.resolve()}")

if __name__ == "__main__":
    main()
