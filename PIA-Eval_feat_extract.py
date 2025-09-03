import os, json, torch, numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.transforms import Compose, Lambda
from torchvision.transforms.v2 import CenterCrop, Normalize
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)
from pytorchvideo.data import LabeledVideoDataset, UniformClipSampler
from torchvision.io import read_video

# ========= 전역 설정 =========
PIA_EVAL_ROOT = "PIA-Eval"          # normal/, violence/ 가 포함된 루트
OUT_NPY_ROOT  = "PIA-Eval_npy"      # .npy / .meta.json 저장 폴더
OUT_LIST      = "test_PIA-Eval_list.txt"

MODEL_NAME = "x3d_l"
FRAMES_PER_SECOND = 30
MODEL_TRANSFORM_PARAMS = {
    "x3d_xs": {"side_size": 182, "crop_size": 182, "num_frames": 4,  "sampling_rate": 12},
    "x3d_s" : {"side_size": 182, "crop_size": 182, "num_frames": 13, "sampling_rate": 6},
    "x3d_m" : {"side_size": 256, "crop_size": 256, "num_frames": 16, "sampling_rate": 5},
    "x3d_l" : {"side_size": 320, "crop_size": 320, "num_frames": 16, "sampling_rate": 5},
}
P = MODEL_TRANSFORM_PARAMS[MODEL_NAME]
SIDE_SIZE, CROP_SIZE = P["side_size"], P["crop_size"]
NUM_FRAMES, SAMPLING_RATE = P["num_frames"], P["sampling_rate"]
CLIP_DURATION = (NUM_FRAMES * SAMPLING_RATE) / FRAMES_PER_SECOND  # ★ 원본과 동일 계산  :contentReference[oaicite:3]{index=3}

# 정규화(mean/std) – 원본과 동일  :contentReference[oaicite:4]{index=4}
MEAN = [0.45, 0.45, 0.45]
STD  = [0.225, 0.225, 0.225]

# STEAD가 기대하는 출력 특징 텐서 크기 (C,T,H,W)=(192,16,10,10)
OUT_C, OUT_T, OUT_HW = 192, 16, 10

# ========= X3D-L 로드 & 분류기 제거 =========
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.hub.load('facebookresearch/pytorchvideo', MODEL_NAME, pretrained=True)
model = model.eval().to(device)
# 원본처럼 마지막 blocks 제거(분류기 제거) 후 중간 feature 사용  :contentReference[oaicite:5]{index=5}
del model.blocks[-1]

# ========= 원본과 동일한 전처리 Transform 정의 =========
class Permute(torch.nn.Module):
    def __init__(self, dims): super().__init__(); self.dims = dims
    def forward(self, x): return torch.permute(x, self.dims)

# 원본 순서: Subsample → /255 → Permute(CTHW) → Normalize → ShortSideScale → CenterCrop → Permute(TCHW)
# (원본의 ApplyTransformToKey + Compose 구성)  :contentReference[oaicite:6]{index=6}
transform = ApplyTransformToKey(
    key="video",
    transform=Compose([
        UniformTemporalSubsample(NUM_FRAMES),
        Lambda(lambda x: x/255.0),
        Permute((1, 0, 2, 3)),                 # (T,C,H,W) -> (C,T,H,W)
        Normalize(MEAN, STD),
        ShortSideScale(size=SIDE_SIZE),
        CenterCrop((CROP_SIZE, CROP_SIZE)),
        Permute((1, 0, 2, 3)),                 # (C,T,H,W) -> (T,C,H,W)
    ])
)

# ========= 유틸 =========
def _scan_videos(root):
    paths = []
    for cls, lab in [("normal", 0), ("violence", 1)]:
        d = os.path.join(root, cls)
        if not os.path.isdir(d): continue
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith((".mp4",".avi",".mov",".mkv",".webm")):
                vp = os.path.join(d, fn)
                base = os.path.splitext(fn)[0]  # ★ 접두어 없이 원본 basename으로 저장
                paths.append((vp, base, lab))
    return paths

def _ensure_dir(path): os.makedirs(path, exist_ok=True)

def _read_fps_total(video_path):
    try:
        frames, _, info = read_video(video_path, pts_unit="sec")
        fps = int(round(info.get("video_fps", 30)))
        tot = int(frames.shape[0])
    except Exception:
        fps, tot = 30, 0
    return fps, tot

def _finalize_and_save(base, video_path, label, feat_agg):
    """
    feat_agg: torch.Tensor shape (C', t', h', w') – 클립들에 대해 max-aggregate 된 마지막 feature
    저장: (192,16,10,10) .npy + .meta.json
    """
    # (t,h,w) 정규화
    x = feat_agg.unsqueeze(0)  # (1,C',t',h',w')
    x = F.interpolate(x, size=(OUT_T, OUT_HW, OUT_HW), mode="trilinear", align_corners=False)
    # 채널 192로 투영(필요 시) — 분포 보존 위해 ReLU 포함
    if x.shape[1] != OUT_C:
        W = torch.empty((OUT_C, x.shape[1], 1,1,1), device=x.device, dtype=x.dtype)
        torch.nn.init.kaiming_normal_(W, mode='fan_out', nonlinearity='relu')
        x = torch.conv3d(x, W).clamp_min_(0)
    x = x.squeeze(0).cpu().numpy()  # (192,16,10,10)

    npy_path  = os.path.join(OUT_NPY_ROOT, f"{base}.npy")
    meta_path = os.path.join(OUT_NPY_ROOT, f"{base}.meta.json")

    np.save(npy_path, x)
    fps, tot = _read_fps_total(video_path)
    meta = {
        "clip_index_to_frame": [(0, max(0, tot-1))],  # ★ dataset.py가 기대
        "fps": int(fps),
        "total_frame": int(tot)
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return npy_path

# ========= 메인 =========
def main():
    _ensure_dir(OUT_NPY_ROOT)

    # (A) PIA-Eval 스캔 → labeled_video_paths 구성(원본과 동일한 방식)  :contentReference[oaicite:7]{index=7}
    labeled_video_paths = []
    base_to_src = {}
    for vp, base, lab in _scan_videos(PIA_EVAL_ROOT):
        # LabeledVideoDataset는 두 번째 인자로 전달한 dict를 배치에 그대로 꺼내줍니다(원본과 동일 아이디어)
        labeled_video_paths.append((vp, {"label": lab, "npy_name": base}))
        base_to_src[base] = (vp, lab)

    # (B) UniformClipSampler (원본과 동일)  :contentReference[oaicite:8]{index=8}
    dataset = LabeledVideoDataset(
        labeled_video_paths=labeled_video_paths,
        clip_sampler=UniformClipSampler(CLIP_DURATION),
        transform=transform,
        decode_audio=False
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    # (C) 각 비디오별로 클립 피처들을 집계(max-pool across clips) 후 저장
    items = []
    cur_name, cur_feat = None, None
    with tqdm(total=len(labeled_video_paths), desc="Extracting features (x3d_l)") as pbar_vid:
        processed_video_count = 0
        for batch in loader:
            vid   = batch["video"].to(device)          # (B=1, C=3, T=16, H, W) after transform & permutes
            name  = batch["npy_name"][0]               # 우리가 label dict에 넣어둔 필드
            # 모델 forward (마지막 블록 삭제된 상태라 중간 feature 반환)
            with torch.no_grad():
                out = model(vid).detach().cpu()        # (1, C', t', h', w')
            feat = out[0]                               # (C', t', h', w')

            if cur_name is None:
                # 첫 클립
                cur_name, cur_feat = name, feat.clone()
            elif name == cur_name:
                # 같은 비디오의 다음 클립 → clip-wise max aggregation
                cur_feat = torch.maximum(cur_feat, feat)
            else:
                # 비디오가 바뀌면 이전 비디오 저장
                src_path, lab = base_to_src[cur_name]
                npy_path = _finalize_and_save(cur_name, src_path, lab, cur_feat)
                items.append(npy_path)
                processed_video_count += 1
                pbar_vid.update(1)

                # 새로운 비디오로 초기화
                cur_name, cur_feat = name, feat.clone()

        # 마지막 비디오 저장
        if cur_name is not None:
            src_path, lab = base_to_src[cur_name]
            npy_path = _finalize_and_save(cur_name, src_path, lab, cur_feat)
            items.append(npy_path)
            processed_video_count += 1
            pbar_vid.update(1)

    # (D) 리스트 파일 저장 (평가 스크립트가 그대로 사용)  :contentReference[oaicite:9]{index=9}
    with open(OUT_LIST, "w", encoding="utf-8") as f:
        for p in items:
            f.write(p + "\n")

    print(f"[INFO] Done. Saved {processed_video_count} videos to {OUT_NPY_ROOT}")
    print(f"[INFO] List file: {OUT_LIST}")

if __name__ == "__main__":
    main()
