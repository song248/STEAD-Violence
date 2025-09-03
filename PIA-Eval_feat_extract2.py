# PIA-Eval_feat_extract.py — 2초/8프레임, 1초 스트라이드 (수정: Normalize 위치/permute 순서)
import os, json, torch, numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.io import read_video
from torchvision.transforms import Compose, Lambda
from torchvision.transforms.v2 import CenterCrop, Normalize
from pytorchvideo.transforms import ShortSideScale

# ========= 전역 설정 =========
PIA_EVAL_ROOT = "PIA-Eval"          # normal/, violence/ 가 포함된 루트
OUT_NPY_ROOT  = "PIA-Eval_npy"      # .npy / .meta.json 저장 폴더
OUT_LIST      = "test_PIA-Eval_list.txt"

# 슬라이딩 윈도우 파라미터
CLIP_SEC      = 2.0                 # 2초 길이
STRIDE_SEC    = 1.0                 # 1초 스트라이드(겹침 O)
NUM_FRAMES    = 8                   # 클립 당 8프레임 균일 샘플

# 전처리(원본 파이프라인과 동일한 값)
SIDE_SIZE     = 320
CROP_SIZE     = 320
MEAN          = [0.45, 0.45, 0.45]
STD           = [0.225, 0.225, 0.225]

# STEAD 입력 규격
OUT_C, OUT_T, OUT_HW = 192, 16, 10

# ========= X3D-L 로드 & 분류기 제거 =========
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_l', pretrained=True)
del model.blocks[-1]
model = model.eval().to(device)

# ========= 전처리 Transform =========
class Permute(torch.nn.Module):
    def __init__(self, dims): super().__init__(); self.dims = dims
    def forward(self, x): return torch.permute(x, self.dims)

# ❗핵심 수정:
# - Normalize는 (T,C,H,W)에서 수행해야 채널=3에 정상 브로드캐스팅 됨
# - ShortSideScale / CenterCrop만 (C,T,H,W)에서 수행
preprocess = Compose([
    Lambda(lambda x: x/255.0),              # (T,C,H,W), [0,1]
    Normalize(MEAN, STD),                   # (T,C,H,W)에서 채널=3 기준 정규화
    Permute((1, 0, 2, 3)),                  # (C,T,H,W)로 전환 → 아래 스케일/크롭은 이 형식에서
    ShortSideScale(size=SIDE_SIZE),
    CenterCrop((CROP_SIZE, CROP_SIZE)),
    Permute((1, 0, 2, 3)),                  # 다시 (T,C,H,W)
])

# ========= 유틸 =========
def _ensure_dir(p): os.makedirs(p, exist_ok=True)

def _scan_videos(root):
    paths = []
    for cls, lab in [("normal", 0), ("violence", 1)]:
        d = os.path.join(root, cls)
        if not os.path.isdir(d): continue
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith((".mp4",".avi",".mov",".mkv",".webm")):
                vp = os.path.join(d, fn)
                base = os.path.splitext(fn)[0]
                paths.append((vp, base, lab))
    return paths

def _uniform_idx(s, e, num, max_idx):
    s = max(0, min(s, max_idx)); e = max(s+1, min(e, max_idx+1))
    idx = np.linspace(s, e-1, num=num)
    idx = np.clip(np.round(idx).astype(int), 0, max_idx)
    return idx

def _read_video_all(video_path):
    frames, _, info = read_video(video_path, pts_unit="sec")   # (T,H,W,C)
    fps = float(info.get("video_fps", 30.0))
    tot = int(frames.shape[0])
    return frames, fps, tot

def _clip_generator(tot_frames, fps, clip_sec, stride_sec):
    if tot_frames == 0: return
    step = max(1, int(round(stride_sec * fps)))
    win  = max(1, int(round(clip_sec   * fps)))
    for s in range(0, tot_frames, step):
        e = s + win
        if s >= tot_frames: break
        yield s, min(e, tot_frames)

@torch.no_grad()
def _forward_feature(clip_1xcxthw):
    out = model(clip_1xcxthw.to(device)).detach().cpu()        # (1, C', t', h', w')
    return out[0]                                              # (C', t', h', w')

def _finalize_and_save(base, video_path, feat_agg):
    x = feat_agg.unsqueeze(0)                                   # (1,C',t',h',w')
    x = F.interpolate(x, size=(OUT_T, OUT_HW, OUT_HW), mode="trilinear", align_corners=False)
    if x.shape[1] != OUT_C:
        W = torch.empty((OUT_C, x.shape[1], 1,1,1), device=x.device, dtype=x.dtype)
        torch.nn.init.kaiming_normal_(W, mode='fan_out', nonlinearity='relu')
        x = torch.conv3d(x, W).clamp_min_(0)
    x = x.squeeze(0).numpy()                                    # (192,16,10,10)

    npy_path  = os.path.join(OUT_NPY_ROOT, f"{base}.npy")
    meta_path = os.path.join(OUT_NPY_ROOT, f"{base}.meta.json")
    np.save(npy_path, x)

    try:
        _, fps, tot = _read_video_all(video_path)
    except Exception:
        fps, tot = 30.0, 0
    meta = {
        "clip_index_to_frame": [(0, max(0, tot-1))],
        "fps": int(round(fps)),
        "total_frame": int(tot)
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return npy_path

# ========= 메인 =========
def main():
    _ensure_dir(OUT_NPY_ROOT)
    items = []
    vids = _scan_videos(PIA_EVAL_ROOT)

    with tqdm(total=len(vids), desc="Extracting (2s/8f, stride 1s)") as pbar:
        for video_path, base, _lab in vids:
            try:
                frames, fps, tot = _read_video_all(video_path)       # (T,H,W,C), tot
                if tot == 0:
                    tqdm.write(f"[WARN] empty video: {video_path}")
                    pbar.update(1); continue

                cur_feat = None
                for s, e in _clip_generator(tot, fps, CLIP_SEC, STRIDE_SEC):
                    idx = _uniform_idx(s, e, NUM_FRAMES, tot-1)
                    clip = frames[idx]                               # (8,H,W,C)
                    clip = clip.float().permute(0,3,1,2)             # (T,C,H,W)=(8,3,H,W)

                    # ✅ 수정된 전처리: Normalize는 (T,C,H,W)에서, Resize/Crop은 (C,T,H,W)에서
                    clip = preprocess(clip)                          # (T,C,H,W)
                    clip = clip.permute(1,0,2,3).unsqueeze(0)        # (1,C,T,H,W)=(1,3,8,*,*)

                    feat = _forward_feature(clip)                    # (C', t', h', w')
                    cur_feat = feat.clone() if cur_feat is None else torch.maximum(cur_feat, feat)

                npy_path = _finalize_and_save(base, video_path, cur_feat)
                items.append(npy_path)

            except Exception as e:
                tqdm.write(f"[WARN] fail: {video_path} ({e})")
            finally:
                pbar.update(1)

    with open(OUT_LIST, "w", encoding="utf-8") as f:
        for p in items: f.write(p + "\n")

    print(f"[INFO] Done. Saved {len(items)} videos to {OUT_NPY_ROOT}")
    print(f"[INFO] List file: {OUT_LIST}")

if __name__ == "__main__":
    main()
