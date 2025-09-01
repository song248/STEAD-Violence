# experiments2.py — Streaming(overlap) + per-file tqdm, x3d_l fixed, safe pin_memory, no .npy intermediate
import os
import glob
from pathlib import Path
from typing import List, Tuple, Deque
from collections import deque

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torchvision.io import VideoReader
from tqdm import tqdm
import cv2  # fps, total frames meta

# =========================
# 사용자 옵션 (argparse 미사용)
# =========================
VIDEO_DIR = "hf-violence/video"
CKPT_PATH = "saved_models/GJ_finetune.pkl"
CLIP_DURATION_SEC = 2
FRAMES_PER_CLIP   = 8      # 비교 실험 시 8/16 등 자유롭게 변경
STRIDE_SEC        = 1     # 겹침 유지
BATCH_SIZE        = 8       # 소배치 추론
THRESHOLD         = 0.5
USE_AMP           = False   # 재현성 우선(속도 필요하면 True로 바꾸세요)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_ROOT = f"hf-violence/predict(FT)-{CLIP_DURATION_SEC}sec-{FRAMES_PER_CLIP}fps-{STRIDE_SEC}"
os.makedirs(OUT_ROOT, exist_ok=True)

torch.backends.cudnn.benchmark = True  # 고정 해상도 가정 시 커널 튜닝

# =========================
# 전처리 (feat_extractor.py와 동일 흐름)
# =========================
MEAN = [0.45, 0.45, 0.45]
STD  = [0.225, 0.225, 0.225]
SHORT_SIDE = 320
CROP_SIZE  = 320

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

def uniform_temporal_subsample_thwc(video_thwc: torch.Tensor, num_samples: int) -> torch.Tensor:
    # (T,H,W,C) → 균일 샘플링
    assert video_thwc.ndim == 4 and video_thwc.shape[-1] == 3
    t = video_thwc.shape[0]
    if t == num_samples or t == 0:
        return video_thwc
    idx = torch.linspace(0, max(t - 1, 0), num_samples, device=video_thwc.device)
    idx = idx.round().long().clamp(0, t - 1)
    return video_thwc.index_select(0, idx)

def preprocess_clip_from_thwc(video_thwc: torch.Tensor) -> torch.Tensor:
    """
    입력: (T,H,W,C) uint8
    출력: (C,T,S,S) float32 normalized
    """
    v = uniform_temporal_subsample_thwc(video_thwc, FRAMES_PER_CLIP)
    v = v.to(torch.float32) / 255.0
    v = v.permute(0, 3, 1, 2).contiguous()      # (T,C,H,W)
    v = short_side_resize_tc_hw(v, SHORT_SIDE)  # (T,C,H',W')
    v = center_crop_tc_hw(v, CROP_SIZE)         # (T,C,S,S)
    v = normalize_tc_hw(v, MEAN, STD)           # (T,C,S,S)
    v = v.permute(1, 0, 2, 3).contiguous()      # (C,T,S,S)
    return v

# =========================
# X3D feature extractor — x3d_l 고정 (원래 파이프라인과 동일)
# =========================
def load_x3d_feature_extractor():
    model = torch.hub.load("facebookresearch/pytorchvideo", "x3d_l", pretrained=True)  # 고정
    del model.blocks[-1]  # 분류헤드 제거
    model.eval().to(DEVICE)
    return model

# =========================
# tiny 분류 모델 로딩 (dims=(32,32), depths=(1,1), ff_mult=1)
# =========================
from model import Model

def load_tiny_model(ckpt_path: str, device: torch.device) -> Model:
    model = Model(dropout=0.0, attn_dropout=0.0, ff_mult=1, dims=(32,32), depths=(1,1))
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    return model

# =========================
# 메타(fps, 총 프레임)
# =========================
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
    hop = int(round(stride_sec * fps))
    if clip_len <= 0 or hop <= 0:
        return spans
    for start in range(0, max(1, num_frames - clip_len + 1), hop):
        end = min(num_frames - 1, start + clip_len - 1)
        spans.append((start, end))
        if end == num_frames - 1:
            break
    return spans

# =========================
# 안전한 pin_memory / GPU 전송 유틸
# =========================
def safe_pin_memory(t: torch.Tensor) -> torch.Tensor:
    if DEVICE.type == "cuda" and torch.cuda.is_available() and t.device.type == "cpu":
        try:
            return t.pin_memory()
        except RuntimeError:
            return t
    return t

def safe_to_device(t: torch.Tensor) -> torch.Tensor:
    if DEVICE.type == "cuda":
        non_blocking = (t.device.type == "cpu" and getattr(t, "is_pinned", lambda: False)())
        return t.to(DEVICE, non_blocking=non_blocking)
    return t

# =========================
# 스트리밍 + 슬라이딩 윈도우 추론 (stride=0.5 유지)
# =========================
@torch.no_grad()
def infer_video_to_frame_probs_streaming(x3d, clf_model, video_path: str):
    fps, num_frames = get_video_meta_cv2(video_path)
    clip_len = int(round(CLIP_DURATION_SEC * fps))
    hop = int(round(STRIDE_SEC * fps))
    spans = compute_clip_spans(num_frames, fps, CLIP_DURATION_SEC, STRIDE_SEC)

    # 파일별 진행도 바
    pbar = tqdm(total=len(spans), desc=os.path.basename(video_path), unit="clip", leave=False)

    # VideoReader 1회 오픈
    vr = VideoReader(video_path, "video")

    # ring buffer: 최근 프레임 (H,W,C) 텐서를 보관
    ring: Deque[torch.Tensor] = deque()
    produced = 0
    next_start = 0
    clip_probs = np.zeros((len(spans),), dtype=np.float32)

    frame_idx = 0
    batch_feats, batch_idx = [], []

    for frame in vr:
        img_hwc = frame["data"]  # uint8, (H,W,C)
        if img_hwc.ndim == 3 and img_hwc.shape[0] in (1, 3):  # (C,H,W) 케이스 방지
            img_hwc = img_hwc.permute(1, 2, 0).contiguous()
        ring.append(img_hwc.cpu())
        frame_idx += 1

        # 메모리 상수 유지: next_start 이전 프레임은 drop
        while len(ring) > (frame_idx - next_start):
            ring.popleft()

        # 윈도우가 준비되면 생성
        while (frame_idx - next_start) >= clip_len and produced < len(spans):
            current_len = len(ring)
            start_offset = current_len - (frame_idx - next_start)
            window_list = list(ring)[start_offset : start_offset + clip_len]
            clip_thwc = torch.stack(window_list, dim=0)             # (T,H,W,C)
            clip_cthw = preprocess_clip_from_thwc(clip_thwc)        # (C,T,S,S)

            # X3D → feature (x3d_l 고정)
            if USE_AMP and DEVICE.type == "cuda":
                with torch.cuda.amp.autocast():
                    x3d_out = x3d(clip_cthw.unsqueeze(0).to(DEVICE, non_blocking=True))
            else:
                x3d_out = x3d(clip_cthw.unsqueeze(0).to(DEVICE, non_blocking=True))

            # 가능하면 에러를 즉시 표면화
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            feat = x3d_out.detach().cpu()[0]                        # (C,T,H,W)

            # 분류 소배치
            batch_feats.append(feat)
            batch_idx.append(produced)

            # 배치 추론
            if len(batch_feats) >= BATCH_SIZE or produced == len(spans) - 1:
                feats_tensor = torch.stack(batch_feats, dim=0)      # (B,C,T,H,W)
                feats_tensor = safe_pin_memory(feats_tensor)
                if USE_AMP and DEVICE.type == "cuda":
                    with torch.cuda.amp.autocast():
                        feats_tensor = safe_to_device(feats_tensor)
                        logits, _ = clf_model(feats_tensor)
                        probs_t = torch.sigmoid(logits).squeeze(-1)
                else:
                    feats_tensor = safe_to_device(feats_tensor)
                    logits, _ = clf_model(feats_tensor)
                    probs_t = torch.sigmoid(logits).squeeze(-1)

                probs = probs_t.detach().cpu().numpy().astype(np.float32).reshape(-1)  # 0-D 방지
                for bi, p in zip(batch_idx, probs):
                    clip_probs[bi] = p
                batch_feats.clear(); batch_idx.clear()
                del feats_tensor, probs_t, logits
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

            produced += 1
            pbar.update(1)

            next_start += hop
            while len(ring) > (frame_idx - next_start):
                ring.popleft()

            if produced >= len(spans):
                break

    pbar.close()

    # 겹침 평균 → 프레임별 확률/라벨
    sums = np.zeros((num_frames,), dtype=np.float64)
    cnts = np.zeros((num_frames,), dtype=np.int32)
    for (s, e), p in zip(spans, clip_probs):
        sums[s:e+1] += float(p)
        cnts[s:e+1] += 1
    with np.errstate(divide='ignore', invalid='ignore'):
        frame_probs = np.where(cnts > 0, sums / np.maximum(cnts, 1), 0.0).astype(np.float32)
    frame_labels = (frame_probs >= THRESHOLD).astype(np.int64)
    return frame_probs, frame_labels

# =========================
# 메인
# =========================
def main():
    x3d = load_x3d_feature_extractor()          # x3d_l 고정
    clf  = load_tiny_model(CKPT_PATH, DEVICE)   # tiny 설정 로드

    video_paths = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")))
    if len(video_paths) == 0:
        raise SystemExit(f"No .mp4 videos found under {VIDEO_DIR}")

    for vp in video_paths:  # 파일별 tqdm만 출력
        base = os.path.splitext(os.path.basename(vp))[0]
        out_csv = os.path.join(OUT_ROOT, f"{base}.csv")
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)

        frame_probs, frame_labels = infer_video_to_frame_probs_streaming(x3d, clf, vp)

        frames = np.arange(len(frame_probs), dtype=np.int64)
        df = pd.DataFrame({
            "frame": frames,
            "violence": frame_labels,
            "prob": frame_probs,
        })
        df.to_csv(out_csv, index=False)

    print(f"[DONE] Saved CSVs under: {Path(OUT_ROOT).resolve()}")

if __name__ == "__main__":
    main()
