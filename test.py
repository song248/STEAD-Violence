# test.py
from torch.utils.data import DataLoader
import option
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from tqdm import tqdm
from model import Model
from dataset import Dataset
from torchinfo import summary
import umap
import numpy as np
import sys
import os

# =========================
# 체크포인트 경로/이름 (직접 수정)
# =========================
MODEL_LOCATION   = 'ckpt/0.0002_16_tiny/'   # 예: 'ckpt/0.0002_16_tiny/'
MODEL_NAME       = 'model29-x3d'            # 예: 'model29-x3d'
MODEL_EXTENSION  = '.pkl'                   # 예: '.pkl'

def test(dataloader, model, args, device='cuda', name="testing", main=False):
    model.to(device)
    plt.clf()
    with torch.no_grad():
        model.eval()
        pred = []
        labels = []
        feats = []

        for _, inputs in tqdm(enumerate(dataloader)):
            # inputs: (features, label)
            labels += inputs[1].cpu().detach().tolist()
            x = inputs[0].to(device)  # (B, C, T, H, W)
            scores, feat = model(x)
            scores = torch.nn.Sigmoid()(scores).squeeze()
            pred_ = scores.cpu().detach().tolist()
            feats += feat.cpu().detach().tolist()
            pred += pred_

        # AUC 지표
        fpr, tpr, _ = roc_curve(labels, pred)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(labels, pred)
        pr_auc = auc(recall, precision)
        print('pr_auc : ' + str(pr_auc))
        print('roc_auc : ' + str(roc_auc))

        # 임베딩 시각화(선택)
        if main:
            feats = np.array(feats)
            reducer = umap.UMAP()
            reduced = reducer.fit_transform(feats)
            labels_np = np.array(labels)
            plt.figure()
            plt.scatter(reduced[labels_np == 0, 0], reduced[labels_np == 0, 1],
                        c='tab:blue', label='Normal', marker='o')
            plt.scatter(reduced[labels_np == 1, 0], reduced[labels_np == 1, 1],
                        c='tab:red', label='Anomaly', marker='*')
            plt.title('UMAP Embedding of Video Features')
            plt.xlabel('UMAP Dimension 1')
            plt.ylabel('UMAP Dimension 2')
            plt.legend()
            plt.savefig(name + "_embed.png", bbox_inches='tight')
            plt.close()

        return roc_auc, pr_auc

def load_checkpoint(model, ckpt_path, device):
    """일반/DP 모델 모두 호환 로딩"""
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']

    # 'module.' 접두어 정리
    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            if k.startswith('module.'):
                new_state[k[len('module.'):]] = v
            else:
                new_state[k] = v
        state = new_state

    # 엄격도 완화(계층명/버전 차이 대비)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] missing keys: {missing}")
    if unexpected:
        print(f"[WARN] unexpected keys: {unexpected}")
    return model

if __name__ == '__main__':
    args = option.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 구성: 기존 로직 유지
    if args.model_arch == 'base':
        model = Model()
    elif args.model_arch == 'fast' or args.model_arch == 'tiny':
        model = Model(ff_mult=1, dims=(32, 32), depths=(1, 1))
    else:
        print('Model architecture not recognized')
        sys.exit(1)

    # 데이터로더
    test_loader = DataLoader(Dataset(args, test_mode=True),
                             batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=False)

    # 요약(입력 텐서 크기 표기)
    model = model.to(device)
    summary(model, (1, 192, 16, 10, 10))

    # 체크포인트 로드
    ckpt_path = os.path.join(MODEL_LOCATION, MODEL_NAME + MODEL_EXTENSION)
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    model = load_checkpoint(model, ckpt_path, device)

    # 테스트 실행 (이름은 MODEL_NAME으로 저장/표시)
    roc_auc, pr_auc = test(test_loader, model, args, device, name=MODEL_NAME, main=True)
