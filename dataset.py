import torch.utils.data as data
import numpy as np, torch, random, os, json
import option
args = option.parse_args()

def enumerate_clip_indices(npy_path: str):
    arr = np.load(npy_path, mmap_mode="r", allow_pickle=True)
    if arr.ndim == 5:   # (num_clips, C, T, H, W)
        return list(range(arr.shape[0]))
    elif arr.ndim == 4: # (C, T, H, W) -> 단일 클립
        return [0]
    else:
        raise ValueError(f"Unexpected array shape {arr.shape} in {npy_path}")

def load_meta(meta_path: str):
    with open(meta_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    return m["clip_index_to_frame"], int(m["fps"]), int(m["total_frame"])

def load_json_intervals(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        j = json.load(f)
    intervals = []
    for _, v in j.get("clips", {}).items():
        ts = v.get("timestamp", None)
        if ts and len(ts) == 2:
            intervals.append((int(ts[0]), int(ts[1])))
    return intervals

def iou_1d(a, b):
    s1, e1 = a; s2, e2 = b
    inter = max(0, min(e1, e2) - max(s1, s2) + 1)
    if inter == 0:
        return 0.0
    union = (e1 - s1 + 1) + (e2 - s2 + 1) - inter
    return inter / union

class Dataset(data.Dataset):
    def __init__(self, args, test_mode: bool = False, seed: int = 2025):
        self.args = args
        self.test_mode = test_mode
        self.rng = random.Random(seed)

        list_file = args.test_rgb_list if test_mode else args.rgb_list
        with open(list_file, "r") as f:
            self.files = [ln.strip() for ln in f if ln.strip()]

        self.all_items = []  # (npy_path, clip_idx, label)
        for npy_path in self.files:
            base = os.path.splitext(os.path.basename(npy_path))[0]
            meta_path = os.path.join(args.npy_root, f"{base}.meta.json")
            json_path = os.path.join(args.data_root, f"{base}.json")

            clip_spans, fps, total_frame = load_meta(meta_path)
            intervals = load_json_intervals(json_path)

            for ci in enumerate_clip_indices(npy_path):
                cs, ce = clip_spans[ci]
                label = 0.0
                for (as_, ae_) in intervals:
                    iou = iou_1d([cs, ce], [as_, ae_])
                    if iou >= self.args.iou_th:  # 겹침만 보려면 0.0
                        label = 1.0
                        break
                self.all_items.append((npy_path, ci, label))

        if not self.test_mode:
            self.normal_indices = [i for i, it in enumerate(self.all_items) if it[2] == 0.0]
            self.abnorm_indices = [i for i, it in enumerate(self.all_items) if it[2] == 1.0]
            if len(self.normal_indices) == 0 or len(self.abnorm_indices) == 0:
                raise RuntimeError("정상/이상 클립이 모두 필요합니다. JSON 라벨/IoU 임계값을 확인하세요.")
            self._reset_pools()

    def _reset_pools(self):
        self.n_pool = self.normal_indices.copy()
        self.a_pool = self.abnorm_indices.copy()
        self.rng.shuffle(self.n_pool)
        self.rng.shuffle(self.a_pool)
        self.num_pairs = min(len(self.n_pool), len(self.a_pool))

    def __len__(self):
        if self.test_mode:
            return len(self.all_items)
        else:
            return self.num_pairs

    def _load_clip(self, path: str, clip_idx: int):
        arr = np.load(path, allow_pickle=True)
        if arr.ndim == 5:
            feat = arr[clip_idx]  # (C,T,H,W)
        elif arr.ndim == 4:
            assert clip_idx == 0
            feat = arr
        else:
            raise ValueError(f"Unexpected array shape {arr.shape} in {path}")
        return np.asarray(feat, dtype=np.float32)

    def __getitem__(self, index):
        if not self.test_mode:
            if len(self.n_pool) == 0 or len(self.a_pool) == 0:
                self._reset_pools()
            n_idx = self.n_pool.pop()
            a_idx = self.a_pool.pop()
            n_path, n_clip, _ = self.all_items[n_idx]
            a_path, a_clip, _ = self.all_items[a_idx]
            nfeat = self._load_clip(n_path, n_clip)
            afeat = self._load_clip(a_path, a_clip)
            nlabel = 0.0
            alabel = 1.0
            return nfeat, nlabel, afeat, alabel
        else:
            p, ci, lab = self.all_items[index]
            feat = self._load_clip(p, ci)
            return feat, float(lab)
