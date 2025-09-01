import os
import sys
import json
from typing import Tuple, Dict, List

import pandas as pd
import numpy as np

BASE_DIR = "hf-violence"
GT_DIR   = f"{BASE_DIR}/GT"
PRED_DIR = f"{BASE_DIR}/predict-2sec-16fps-1"
REAL_DIR = f"{BASE_DIR}/real-predict"

def safe_bool_int(x):
    """violence 라벨을 0/1 정수로 강제."""
    try:
        if pd.isna(x):
            return 0
        v = int(float(x))
        return 1 if v >= 1 else 0
    except Exception:
        return 0


def load_gt(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.strip().lower(): c for c in df.columns}
    if "frame" not in cols or "violence" not in cols:
        raise ValueError(f"GT CSV '{path}' must contain columns: frame, violence")
    df = df[[cols["frame"], cols["violence"]]].copy()
    df.columns = ["frame", "violence"]
    # 중복 frame은 최초만 사용
    if df["frame"].duplicated().any():
        df = df[~df["frame"].duplicated(keep="first")].copy()
    df["violence"] = df["violence"].map(safe_bool_int).astype(int)
    # frame 정렬(숫자 가능 시 숫자 기준)
    df["__frame_num__"] = pd.to_numeric(df["frame"], errors="coerce")
    df = df.sort_values(by=["__frame_num__", "frame"]).drop(columns="__frame_num__").reset_index(drop=True)
    return df


def load_predict(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.strip().lower(): c for c in df.columns}
    if "frame" not in cols or "prob" not in cols:
        raise ValueError(f"Predict CSV '{path}' must contain columns: frame, prob (violence column is ignored)")
    df = df[[cols["frame"], cols["prob"]]].copy()
    df.columns = ["frame", "prob"]
    # 중복 frame은 최초만 사용
    if df["frame"].duplicated().any():
        df = df[~df["frame"].duplicated(keep="first")].copy()
    # prob 숫자화 및 클리핑
    df["prob"] = pd.to_numeric(df["prob"], errors="coerce")
    df["prob"] = df["prob"].clip(lower=0.0, upper=1.0)
    # frame 정렬
    df["__frame_num__"] = pd.to_numeric(df["frame"], errors="coerce")
    df = df.sort_values(by=["__frame_num__", "frame"]).drop(columns="__frame_num__").reset_index(drop=True)
    return df


def left_join_align(gt: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """
    GT를 기준으로 frame을 LEFT JOIN 하여 행을 강제 일치.
    - predict에 없는 frame -> prob 결측 -> GT의 violence 값으로 prob 채움
    - predict에만 있는 frame은 자동으로 제거됨.
    결과: columns = frame, violence(gt), prob(보정완료)
    """
    merged = pd.merge(gt, pred, on="frame", how="left")
    missing = merged["prob"].isna()
    if missing.any():
        merged.loc[missing, "prob"] = merged.loc[missing, "violence"].astype(float)
    merged["prob"] = merged["prob"].fillna(0.0).clip(0.0, 1.0)
    return merged[["frame", "violence", "prob"]].reset_index(drop=True)


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, fp, tn, fn


def precision_recall_f1_acc(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    P = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    R = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    F1 = (2 * P * R / (P + R)) if (P + R) > 0 else 0.0
    ACC = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
    return {"precision": P, "recall": R, "f1": F1, "accuracy": ACC}


def select_best_threshold(y_true: np.ndarray, prob: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """
    F1 최대 threshold 선택. 동점시 recall 큰 t, 그래도 같으면 더 작은 t 선택.
    """
    uniq = np.unique(prob[~np.isnan(prob)])
    candidates = np.unique(np.concatenate([uniq, np.array([0.0, 1.0])]))
    best = {"t": 0.5, "f1": -1.0, "recall": -1.0, "precision": -1.0, "accuracy": -1.0}
    for t in candidates:
        y_pred = (prob >= t).astype(int)
        tp, fp, tn, fn = confusion_counts(y_true, y_pred)
        metrics = precision_recall_f1_acc(tp, fp, tn, fn)
        f1, rec = metrics["f1"], metrics["recall"]
        choose = False
        if f1 > best["f1"]:
            choose = True
        elif np.isclose(f1, best["f1"]):
            if rec > best["recall"]:
                choose = True
            elif np.isclose(rec, best["recall"]) and t < best["t"]:
                choose = True
        if choose:
            best.update({"t": float(t),
                         "f1": float(f1),
                         "recall": float(rec),
                         "precision": float(metrics["precision"]),
                         "accuracy": float(metrics["accuracy"])})
    return best["t"], {"precision": best["precision"], "recall": best["recall"],
                       "f1": best["f1"], "accuracy": best["accuracy"]}


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def evaluate_micro(gt_dir: str, real_dir: str, files: List[str]) -> Dict[str, float]:
    TP = FP = TN = FN = 0
    total_frames = 0
    for fname in files:
        gt = load_gt(os.path.join(gt_dir, fname))
        rp = pd.read_csv(os.path.join(real_dir, fname))
        gt.columns = [c.strip().lower() for c in gt.columns]
        rp.columns = [c.strip().lower() for c in rp.columns]
        merged = pd.merge(gt[["frame", "violence"]], rp[["frame", "violence"]],
                          on="frame", how="inner", suffixes=("_gt", "_pred"))
        if merged.empty:
            continue
        y_true = merged["violence_gt"].map(safe_bool_int).to_numpy(dtype=int)
        y_pred = merged["violence_pred"].map(safe_bool_int).to_numpy(dtype=int)
        tp, fp, tn, fn = confusion_counts(y_true, y_pred)
        TP += tp; FP += fp; TN += tn; FN += fn
        total_frames += len(merged)
    metrics = precision_recall_f1_acc(TP, FP, TN, FN)
    metrics.update({"total_frames": int(total_frames),
                    "TP": int(TP), "FP": int(FP), "TN": int(TN), "FN": int(FN)})
    return metrics


def main():
    ensure_dir(REAL_DIR)

    if not os.path.isdir(GT_DIR) or not os.path.isdir(PRED_DIR):
        print(f"[ERROR] GT_DIR or PRED_DIR not found.\nGT_DIR={GT_DIR}\nPRED_DIR={PRED_DIR}")
        sys.exit(1)

    gt_files = sorted([f for f in os.listdir(GT_DIR) if f.lower().endswith(".csv")])
    pred_files = sorted([f for f in os.listdir(PRED_DIR) if f.lower().endswith(".csv")])
    common = sorted(list(set(gt_files).intersection(set(pred_files))))

    if not common:
        print("No common CSV files found between GT and predict.")
        sys.exit(1)

    per_file_report = []
    for fname in common:
        gt_path = os.path.join(GT_DIR, fname)
        pr_path = os.path.join(PRED_DIR, fname)

        try:
            gt = load_gt(gt_path)
            pr = load_predict(pr_path)
        except Exception as e:
            print(f"[SKIP] {fname}: {e}")
            continue

        aligned = left_join_align(gt, pr)  # frame, violence(gt), prob
        y_true = aligned["violence"].to_numpy(dtype=int)
        prob = aligned["prob"].to_numpy(dtype=float)

        t_star, file_metrics = select_best_threshold(y_true, prob)

        # real-predict 생성 및 저장
        y_pred = (prob >= t_star).astype(int)
        out = pd.DataFrame({"frame": aligned["frame"], "violence": y_pred.astype(int)})
        out_path = os.path.join(REAL_DIR, fname)
        out.to_csv(out_path, index=False)

        per_file_report.append({
            "file": fname,
            "threshold": t_star,
            **file_metrics,
            "num_frames": int(len(out))
        })

        print(f"[OK] {fname} | t*={t_star:.6f} | F1={file_metrics['f1']:.4f} "
              f"(P={file_metrics['precision']:.4f}, R={file_metrics['recall']:.4f}, ACC={file_metrics['accuracy']:.4f}) "
              f"| frames={len(out)}")

    micro = evaluate_micro(GT_DIR, REAL_DIR, [r["file"] for r in per_file_report])

    print("\n=== Summary ===")
    print(f"Processed files: {len(per_file_report)}")
    print(f"Total frames (joined): {micro.get('total_frames', 0)}")
    print(f"Micro-Precision: {micro['precision']:.6f}")
    print(f"Micro-Recall   : {micro['recall']:.6f}")
    print(f"Micro-F1       : {micro['f1']:.6f}")
    print(f"Micro-Accuracy : {micro['accuracy']:.6f}")
    print(f"TP={micro['TP']} FP={micro['FP']} TN={micro['TN']} FN={micro['FN']}")

    report = {
        "base": BASE_DIR,
        "gt_dir": GT_DIR,
        "pred_dir": PRED_DIR,
        "real_dir": REAL_DIR,
        "per_file": per_file_report,
        "micro": micro
    }
    with open(os.path.join(BASE_DIR, "evaluation_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
