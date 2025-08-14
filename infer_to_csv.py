import os
import glob
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
from torch import nn

from model import Model

def load_model(ckpt_path: str, device: torch.device):
    model = Model(dropout=0.0, attn_dropout=0.0)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    return model

def npy_to_clips(npy_path: str):
    arr = np.load(npy_path, allow_pickle=True)
    if arr.ndim == 5:
        return arr  # (N, C, T, H, W)
    elif arr.ndim == 4:
        return arr[None, ...]  # (1, C, T, H, W)
    else:
        raise ValueError(f"Unexpected npy shape {arr.shape} in {npy_path}")

def infer_probs_for_file(model: nn.Module,
                         device: torch.device,
                         npy_path: str,
                         batch_size: int = 64):
    clips = npy_to_clips(npy_path)  # (N, C, T, H, W)
    N = clips.shape[0]
    probs = np.zeros((N,), dtype=np.float32)

    with torch.no_grad():
        for start in tqdm(range(0, N, batch_size),
                          total=(N + batch_size - 1)//batch_size,
                          desc=f"Infer clips: {Path(npy_path).name}",
                          leave=False):
            end = min(start + batch_size, N)
            batch = torch.from_numpy(clips[start:end]).to(device).float()  # (B,C,T,H,W)
            logits, _ = model(batch)  # (B, 1)
            p = torch.sigmoid(logits).squeeze(-1).detach().cpu().numpy()  # (B,)
            probs[start:end] = p.astype(np.float32)

    return probs

def probs_to_frame_csv(probs: np.ndarray,
                       fps: int,
                       out_csv_path: str):
    num_clips = len(probs)
    num_frames = int(num_clips * fps)
    prob_per_frame = np.repeat(probs, repeats=fps).astype(np.float32)
    frames = np.arange(num_frames, dtype=np.int64)

    df = pd.DataFrame({"frame": frames, "prob": prob_per_frame})
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    df.to_csv(out_csv_path, index=False)

def main():
    ap = argparse.ArgumentParser(description="Infer clip probs and write per-frame CSVs")
    ap.add_argument("--ckpt", type=str, default="saved_models/my_video_finetune.pkl",
                    help="model checkpoint (.pkl)")
    ap.add_argument("--in_root", type=str, default="hf_npy-1sec-10fps",
                    help="npy root file")
    ap.add_argument("--out_root", type=str, default="hf-violence/predict_hf_1sec_10fps",
                    help="csv save root")
    ap.add_argument("--fps", type=int, default=30,
                    help="video fps")
    ap.add_argument("--batch_size", type=int, default=64,
                    help="inference batch size")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)

    npy_paths = sorted([p for p in glob.glob(os.path.join(args.in_root, "**", "*.npy"), recursive=True)])

    if len(npy_paths) == 0:
        raise SystemExit(f"No .npy files found under {args.in_root}")

    for npy_path in tqdm(npy_paths, desc="Files", unit="file"):
        rel = os.path.relpath(npy_path, args.in_root)           # e.g., normal/vidA.npy
        rel_csv = os.path.splitext(rel)[0] + ".csv"              # e.g., normal/vidA.csv
        out_csv_path = os.path.join(args.out_root, rel_csv)      # e.g., out_root/normal/vidA.csv

        probs = infer_probs_for_file(model, device, npy_path, args.batch_size)
        probs_to_frame_csv(probs, fps=args.fps, out_csv_path=out_csv_path)
    print(f"[DONE] Saved frame-level CSVs under: {Path(args.out_root).resolve()}")

if __name__ == "__main__":
    main()
