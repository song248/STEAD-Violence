import os
import glob
import json
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from pytorchvideo.data import LabeledVideoDataset, UniformClipSampler

# --------------------------------------------------
# 경로 설정
# --------------------------------------------------
INPUT_ROOT  = "GJ_violence"       # GJ_violence/<basename>.mp4 + <basename>.json
OUTPUT_ROOT = "GJ_violence_npy"   # GJ_violence_npy/<basename>.npy + .meta.json
os.makedirs(OUTPUT_ROOT, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# X3D 정규화 파라미터
MEAN = [0.45, 0.45, 0.45]
STD  = [0.225, 0.225, 0.225]

# 클립/샘플링 설정
CLIP_DURATION_SEC = 1.0
TARGET_FRAMES_PER_CLIP = 10
SHORT_SIDE = 320
CROP_SIZE = 320
NUM_WORKERS = 2
BATCH_SIZE = 1

def list_videos(root):
    exts = ("*.mp4","*.avi","*.mov","*.mkv","*.MP4","*.AVI","*.MOV","*.MKV")
    items = []
    for ext in exts:
        for path in glob.glob(os.path.join(root, ext)):
            base = os.path.splitext(os.path.basename(path))[0]
            save_noext = os.path.join(OUTPUT_ROOT, base)
            items.append((path, {"video_label": save_noext, "base": base}))
    return items

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
    new_h = int(round(H * scale))
    new_w = int(round(W * scale))
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
    std = torch.tensor(std, device=tchw.device).view(1, -1, 1, 1)
    return (tchw - mean) / std

def preprocess_clip(video_thwc: torch.Tensor) -> torch.Tensor:
    v = uniform_temporal_subsample_thwc(video_thwc, TARGET_FRAMES_PER_CLIP)
    if v.dtype != torch.float32:
        v = v.to(torch.float32)
    v = v / 255.0
    v = v.permute(0, 3, 1, 2).contiguous()
    v = short_side_resize_tc_hw(v, SHORT_SIDE)
    v = center_crop_tc_hw(v, CROP_SIZE)
    v = normalize_tc_hw(v, MEAN, STD)
    v = v.permute(1, 0, 2, 3).contiguous()  # (C,T,H,W)
    return v

def load_x3d_feature_extractor():
    model = torch.hub.load("facebookresearch/pytorchvideo", "x3d_l", pretrained=True)
    del model.blocks[-1]  # 분류헤드 제거
    model.eval().to(DEVICE)
    return model

def load_json_info(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        j = json.load(f)
    fps = int(j["video_info"]["fps"])
    total_frame = int(j["video_info"]["total_frame"])
    intervals = []
    for _, v in j.get("clips", {}).items():
        ts = v.get("timestamp")
        if isinstance(ts, list) and len(ts) == 2:
            intervals.append((int(ts[0]), int(ts[1])))
    return fps, total_frame, intervals

def flush_with_meta(save_noext, feats_list):
    os.makedirs(os.path.dirname(save_noext), exist_ok=True)
    np.save(f"{save_noext}.npy", np.stack(feats_list) if len(feats_list)>0 else np.empty((0,),dtype=np.float32))

    base_name = os.path.basename(save_noext)
    json_path = os.path.join(INPUT_ROOT, f"{base_name}.json")
    try:
        fps, total_frame, _ = load_json_info(json_path)
    except Exception as e:
        meta = {
            "fps": None, "total_frame": None,
            "stride_sec": CLIP_DURATION_SEC,
            "frames_per_clip": TARGET_FRAMES_PER_CLIP,
            "clip_index_to_frame": None,
            "note": f"Failed to read JSON: {e}"
        }
        with open(f"{save_noext}.meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return

    num_clips = len(feats_list)
    stride_sec = CLIP_DURATION_SEC
    clip_spans = []
    for ci in range(num_clips):
        s = int(round(ci * fps * stride_sec))
        e = int(min(total_frame - 1, round((ci + 1) * fps * stride_sec) - 1))
        clip_spans.append([s, e])

    meta = {
        "fps": fps,
        "total_frame": total_frame,
        "stride_sec": stride_sec,
        "frames_per_clip": TARGET_FRAMES_PER_CLIP,
        "clip_index_to_frame": clip_spans
    }
    with open(f"{save_noext}.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def main():
    labeled_video_paths = list_videos(INPUT_ROOT)
    if len(labeled_video_paths) == 0:
        raise RuntimeError(f"No videos found under {INPUT_ROOT}")

    dataset = LabeledVideoDataset(
        labeled_video_paths=labeled_video_paths,
        clip_sampler=UniformClipSampler(CLIP_DURATION_SEC),
        transform=None,
        decode_audio=False,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True
    )

    model = load_x3d_feature_extractor()

    current_key = None
    feats_buf = []

    pbar = tqdm(loader, desc="Extracting X3D features (1s / 10f)", unit="clip")
    with torch.no_grad():
        for batch in pbar:
            video = batch["video"]
            if video.ndim == 5 and video.shape[0] == 1:
                video = video[0]
            if video.ndim != 4:
                raise RuntimeError(f"Unexpected video tensor shape: {video.shape}")
            if video.shape[-1] != 3:
                if video.shape[0] == 3:
                    video = video.permute(1,2,3,0).contiguous()
                else:
                    raise RuntimeError(f"Expected channels-last, got shape {video.shape}")

            clip_cthw = preprocess_clip(video)
            feats = model(clip_cthw.unsqueeze(0).to(DEVICE)).detach().cpu().numpy()[0]

            key = batch["video_label"][0] if isinstance(batch["video_label"], list) else batch["video_label"]
            base = batch["base"][0] if isinstance(batch["base"], list) else batch.get("base", os.path.basename(key))

            if current_key is None:
                current_key = key

            if key != current_key:
                flush_with_meta(current_key, feats_buf)
                current_key = key
                feats_buf = []

            feats_buf.append(feats)

        if current_key is not None:
            flush_with_meta(current_key, feats_buf)

    print(f"Done. Saved to: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()
