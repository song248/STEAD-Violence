import os, glob
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision.io import read_video
import cv2 

# ===== 사용자 옵션 (argparse 미사용) =====
VIDEO_DIR = "hf-violence/video"
CKPT_PATH = "saved_models/888tiny.pkl"     # tiny로 학습된 ckpt 지정
CLIP_DURATION_SEC = 1.0
FRAMES_PER_CLIP   = 8                          # 학습 기본과 정합
STRIDE_SEC        = 0.5                         # 겹치는 슬라이딩
BATCH_SIZE        = 4                            # 메모리 안전하게 축소
THRESHOLD         = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_ROOT = f"hf-violence/predict-1sec-{FRAMES_PER_CLIP}fps-{STRIDE_SEC}"
os.makedirs(OUT_ROOT, exist_ok=True)

# ===== 전처리 파라미터 (feat_extractor.py 기반) =====
MEAN = [0.45, 0.45, 0.45]
STD  = [0.225, 0.225, 0.225]
SHORT_SIDE = 320
CROP_SIZE  = 320

def uniform_temporal_subsample_thwc(video_thwc: torch.Tensor, num_samples: int) -> torch.Tensor:
    assert video_thwc.ndim == 4 and video_thwc.shape[-1] == 3
    t = video_thwc.shape[0]
    if t == num_samples or t == 0:
        return video_thwc
    idx = torch.linspace(0, max(t - 1, 0), num_samples, device=video_thwc.device)
    idx = idx.round().long().clamp(0, t - 1)
    return video_thwc.index_select(0, idx)

def short_side_resize_tc_hw(tchw: torch.Tensor, short_side: int) -> torch.Tensor:
    T, C, H, W = tchw.shape
    if min(H, W) == short_side:
        return tchw
    scale = float(short_side) / float(min(H, W))
    new_h = int(round(H * scale)); new_w = int(round(W * scale))
    return F.interpolate(tchw, size=(new_h, new_w), mode="bilinear", align_corners=False)

def center_crop_tc_hw(tchw: torch.Tensor, crop_size: int) -> torch.Tensor:
    T, C, H, W = tchw.shape
    if H < crop_size or W < crop_size:
        tchw = F.interpolate(tchw, size=(max(H, crop_size), max(W, crop_size)),
                             mode="bilinear", align_corners=False)
        T, C, H, W = tchw.shape
    top = int((H - crop_size) // 2); left = int((W - crop_size) // 2)
    return tchw[:, :, top: top + crop_size, left: left + crop_size]

def normalize_tc_hw(tchw: torch.Tensor, mean, std) -> torch.Tensor:
    mean = torch.tensor(mean, device=tchw.device).view(1, -1, 1, 1)
    std  = torch.tensor(std,  device=tchw.device).view(1, -1, 1, 1)
    return (tchw - mean) / std

def preprocess_clip_from_thwc(video_thwc: torch.Tensor) -> torch.Tensor:
    v = uniform_temporal_subsample_thwc(video_thwc, FRAMES_PER_CLIP)
    v = v.to(torch.float32) / 255.0
    v = v.permute(0, 3, 1, 2).contiguous()      # (T,C,H,W)
    v = short_side_resize_tc_hw(v, SHORT_SIDE)  # (T,C,H',W')
    v = center_crop_tc_hw(v, CROP_SIZE)         # (T,C,S,S)
    v = normalize_tc_hw(v, MEAN, STD)           # (T,C,S,S)
    v = v.permute(1, 0, 2, 3).contiguous()      # (C,T,S,S)
    return v

# ===== X3D feature extractor (분류헤드 제거) =====
def load_x3d_feature_extractor():
    model = torch.hub.load("facebookresearch/pytorchvideo", "x3d_l", pretrained=True)
    del model.blocks[-1]
    model.eval().to(DEVICE)
    return model

# ===== tiny 분류 모델 =====
from model import Model  # 입력 192채널 특징을 받도록 설계됨【model.py】
def load_tiny_model(ckpt_path: str, device: torch.device) -> Model:
    model = Model(dropout=0.0, attn_dropout=0.0, ff_mult=1, dims=(32,32), depths=(1,1))
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    return model

# ===== 메타만 읽기 (fps, 총 프레임) =====
def get_video_meta_cv2(path: str) -> Tuple[float, int]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return float(fps), total

def compute_clip_spans(num_frames: int, fps: float, clip_sec: float, stride_sec: float) -> List[Tuple[int, int]]:
    spans = []
    clip_len = int(round(clip_sec * fps))
    if clip_len <= 0: return spans
    t = 0.0
    while True:
        start = int(round(t * fps))
        if start >= num_frames: break
        end = min(num_frames - 1, start + clip_len - 1)
        spans.append((start, end))
        if end == num_frames - 1: break
        t += stride_sec
    return spans

@torch.no_grad()
def infer_video_to_frame_probs_streaming(x3d, clf_model, video_path: str):
    # 1) fps / 총 프레임만 가볍게 획득 (메모리 안전)
    fps, num_frames = get_video_meta_cv2(video_path)
    spans = compute_clip_spans(num_frames, fps, CLIP_DURATION_SEC, STRIDE_SEC)

    # 파일별 진행도: 클립 단위
    pbar = tqdm(total=len(spans), desc=os.path.basename(video_path), unit="clip", leave=False)

    # 2) 클립을 "시간창"으로 잘라서 필요한 부분만 디코딩
    clip_probs = np.zeros((len(spans),), dtype=np.float32)
    batch_feats, batch_idx = [], []

    for i, (s, e) in enumerate(spans):
        start_sec = s / fps
        end_sec   = (e + 1) / fps  # inclusive e → exclusive end_sec
        frames_thwc, _, _ = read_video(video_path, start_pts=start_sec, end_pts=end_sec, pts_unit="sec")
        if frames_thwc.dtype != torch.uint8:
            frames_thwc = frames_thwc.to(torch.uint8)

        clip_cthw = preprocess_clip_from_thwc(frames_thwc)                 # (C,T,S,S)
        x3d_out   = x3d(clip_cthw.unsqueeze(0).to(DEVICE))                 # X3D 피처
        feat      = x3d_out.detach().cpu()[0]                              # (C,T,H,W) 특징

        batch_feats.append(feat); batch_idx.append(i)
        # 소배치로 즉시 분류 → 확률만 보관
        if len(batch_feats) >= BATCH_SIZE or i == len(spans) - 1:
            feats_tensor = torch.stack(batch_feats, dim=0).to(DEVICE)      # (B,C,T,H,W)
            logits, _    = clf_model(feats_tensor)
            # probs        = torch.sigmoid(logits).squeeze(-1).detach().cpu().numpy().astype(np.float32)
            probs = torch.sigmoid(logits).squeeze(-1).detach().cpu().numpy().astype(np.float32).reshape(-1)
            for bi, p in zip(batch_idx, probs):
                clip_probs[bi] = p
            batch_feats.clear(); batch_idx.clear()
            # 메모리 즉시 반환
            del feats_tensor, probs, logits
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        pbar.update(1)

    pbar.close()

    # 3) 겹치는 프레임 평균 집계 → 프레임별 확률 & 라벨
    sums = np.zeros((num_frames,), dtype=np.float64)
    cnts = np.zeros((num_frames,), dtype=np.int32)
    for (s, e), p in zip(spans, clip_probs):
        sums[s:e+1] += float(p)
        cnts[s:e+1] += 1
    with np.errstate(divide='ignore', invalid='ignore'):
        frame_probs = np.where(cnts > 0, sums / np.maximum(cnts, 1), 0.0).astype(np.float32)
    frame_labels = (frame_probs >= THRESHOLD).astype(np.int64)
    return frame_probs, frame_labels

def main():
    x3d = load_x3d_feature_extractor()                 # 분류헤드 제거 X3D【feat_extractor.py】
    clf = load_tiny_model(CKPT_PATH, DEVICE)           # tiny 설정 로드【main.py】

    video_paths = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")))
    if len(video_paths) == 0:
        raise SystemExit(f"No .mp4 videos found under {VIDEO_DIR}")

    for vp in video_paths:  # 전체 tqdm 없음 (파일별 tqdm만)
        base = os.path.splitext(os.path.basename(vp))[0]
        out_csv = os.path.join(OUT_ROOT, f"{base}.csv")
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)

        frame_probs, frame_labels = infer_video_to_frame_probs_streaming(x3d, clf, vp)

        frames = np.arange(len(frame_probs), dtype=np.int64)
        pd.DataFrame({"frame": frames, "violence": frame_labels, "prob": frame_probs}).to_csv(out_csv, index=False)

    print(f"[DONE] Saved CSVs under: {Path(OUT_ROOT).resolve()}")

if __name__ == "__main__":
    main()
