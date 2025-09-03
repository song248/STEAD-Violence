# PIA-Eval_evaluate.py  — add per-file CSV report
import os, json, glob, csv, torch, numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from model import Model
from dataset import Dataset

# ===== 전역 설정 =====
PIA_EVAL_ROOT     = "PIA-Eval"       # normal/, violence/ 존재
NPY_ROOT          = "PIA-Eval_npy"            # .npy / .meta.json / (생성되는) .json
DATA_ROOT         = "PIA-Eval_npy"
TEST_RGB_LIST     = "test_PIA-Eval_list.txt"

BATCH_SIZE        = 16
MODEL_ARCH        = "fast"                     # "base" | "fast" | "tiny"
CKPT              = "saved_models/888tiny.pkl"  # 없으면 None
IOU_TH            = 0.0
DROPOUT_RATE      = 0.4
ATTN_DROPOUT_RATE = 0.1
THRESH            = 0.5

# 리포트 저장 경로
REPORT_CSV        = "PIA-Eval_report.csv"

# ---- 유틸: base명 → 라벨(0/1) 매핑을 원본 폴더에서 복구 ----
_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")

def _build_base_to_label_map():
    m = {}
    for lab_name, lab in (("normal", 0), ("violence", 1)):
        d = os.path.join(PIA_EVAL_ROOT, lab_name)
        if not os.path.isdir(d):
            continue
        for ext in _EXTS:
            for vp in glob.glob(os.path.join(d, f"*{ext}")):
                base = os.path.splitext(os.path.basename(vp))[0]
                m[base] = lab
    return m

def _read_total_frame(meta_path):
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        return int(m.get("total_frame", 0))
    except Exception:
        return 0

def _ensure_label_jsons():
    """
    TEST_RGB_LIST의 각 .npy에 대해 DATA_ROOT/{base}.json 존재 확인.
    없으면 violence=1 → 전체 구간 timestamp, normal=0 → 빈 clips 로 생성.
    """
    os.makedirs(DATA_ROOT, exist_ok=True)
    base2lab = _build_base_to_label_map()

    with open(TEST_RGB_LIST, "r", encoding="utf-8") as f:
        npy_paths = [ln.strip() for ln in f if ln.strip()]

    made, skipped = 0, 0
    for npy_path in npy_paths:
        base = os.path.splitext(os.path.basename(npy_path))[0]
        json_path = os.path.join(DATA_ROOT, f"{base}.json")
        if os.path.isfile(json_path):
            skipped += 1
            continue

        meta_path = os.path.join(NPY_ROOT, f"{base}.meta.json")
        tot = _read_total_frame(meta_path)
        lab = base2lab.get(base, 0)

        if lab == 1:
            content = {"clips": {"0": {"timestamp": [0, max(0, tot - 1)]}}}
        else:
            content = {"clips": {}}

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(content, f, ensure_ascii=False, indent=2)
        made += 1

    print(f"[INFO] label JSONs - created: {made}, existed: {skipped}")

def _args_like():
    class A: pass
    a = A()
    a.rgb_list = "unused_for_eval"
    a.test_rgb_list = TEST_RGB_LIST
    a.data_root = DATA_ROOT
    a.npy_root  = NPY_ROOT
    a.iou_th    = IOU_TH
    a.batch_size = BATCH_SIZE
    a.lr, a.max_epoch, a.warmup = 2e-4, 1, 0
    a.model_arch = MODEL_ARCH
    a.dropout_rate = DROPOUT_RATE
    a.attn_dropout_rate = ATTN_DROPOUT_RATE
    a.pretrained_ckpt = None
    a.comment, a.model_name = "eval", "model"
    return a

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # (A) Dataset이 요구하는 json 라벨을 보장(없으면 생성). :contentReference[oaicite:4]{index=4}
    _ensure_label_jsons()

    # (B) Dataset/Loader
    args = _args_like()
    test_ds = Dataset(args, test_mode=True)  # 내부에서 .meta.json / .json 읽어 label 부여 :contentReference[oaicite:5]{index=5}
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             pin_memory=(device.type=="cuda"))

    # (C) 모델
    if MODEL_ARCH == "base":
        model = Model(dropout=DROPOUT_RATE, attn_dropout=ATTN_DROPOUT_RATE)
    else:
        model = Model(dropout=DROPOUT_RATE, attn_dropout=ATTN_DROPOUT_RATE,
                      ff_mult=1, dims=(32,32), depths=(1,1))
    if CKPT:
        state = torch.load(CKPT, map_location=device)
        if isinstance(state, dict) and "state_dict" in state: state = state["state_dict"]
        new_state = {}
        for k, v in state.items():
            new_state[k[7:]] = v if k.startswith("module.") else v
        model.load_state_dict(new_state, strict=False)
    model = model.to(device).eval()

    # (D) 추론 (Dataset 순서 == Loader 순서 → per-file 매칭 OK). :contentReference[oaicite:6]{index=6}
    probs = []
    labels_in_order = []
    with torch.no_grad():
        for feats, lab in test_loader:
            feats = feats.to(device).float()              # (B,192,16,10,10) :contentReference[oaicite:7]{index=7}
            logits, _ = model(feats)
            prob = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            probs.extend(prob.tolist())
            labels_in_order.extend(lab.cpu().numpy().tolist())

    preds = (np.array(probs) >= THRESH).astype(np.int32)
    y_true = np.array(labels_in_order[:len(preds)]).astype(np.int32)

    # (E) 지표 출력
    print(f"Precision: {precision_score(y_true, preds, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_true, preds, zero_division=0):.4f}")
    print(f"Accuracy : {accuracy_score(y_true, preds):.4f}")
    print(f"F1-score : {f1_score(y_true, preds, zero_division=0):.4f}")

    # (F) 리포트 CSV 저장 — 각 항목: 파일/예측/실제/정답여부
    # Dataset은 self.all_items = [(npy_path, clip_idx, label), ...] 를 보유. :contentReference[oaicite:8]{index=8}
    rows = []
    for idx, (npy_path, clip_idx, true_lab) in enumerate(test_ds.all_items[:len(preds)]):
        base = os.path.splitext(os.path.basename(npy_path))[0]
        p = float(probs[idx])
        pred_lab = int(preds[idx])
        true_lab = int(true_lab)
        rows.append({
            "npy_path": npy_path,
            "base": base,
            "clip_idx": clip_idx,
            "pred_prob": p,
            "pred_label": pred_lab,
            "true_label": true_lab,
            "pred_cls": "violence" if pred_lab==1 else "normal",
            "true_cls": "violence" if true_lab==1 else "normal",
            "correct": int(pred_lab==true_lab),
        })

    os.makedirs(os.path.dirname(REPORT_CSV) or ".", exist_ok=True)
    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["npy_path","base","clip_idx","pred_prob","pred_label","true_label","pred_cls","true_cls","correct"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] CSV report saved → {REPORT_CSV}")

if __name__ == "__main__":
    main()
