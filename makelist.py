import argparse, os, glob, random

def collect_npy(root):
    return sorted(glob.glob(os.path.join(root, "*.npy")))

def write_list(paths, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for p in paths:
            f.write(p + "\n")
    print(f"wrote {len(paths)} lines -> {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="GJ_violence_npy")
    ap.add_argument("--train_out", type=str, default="train_rgb_list.txt")
    ap.add_argument("--test_out", type=str, default="test_rgb_list.txt")
    ap.add_argument("--split", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    all_list = collect_npy(args.root)
    assert len(all_list) > 0, f"No npy files under {args.root}"
    rng.shuffle(all_list)
    k = int(len(all_list) * args.split)
    train_files = all_list[:k]
    test_files  = all_list[k:]
    write_list(train_files, args.train_out)
    write_list(test_files, args.test_out)
    print(f"total npy: {len(all_list)} | train: {len(train_files)} | test: {len(test_files)}")

if __name__ == "__main__":
    main()
