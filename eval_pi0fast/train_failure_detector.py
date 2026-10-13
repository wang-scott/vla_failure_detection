"""用 pi0-FAST 每層特徵預測 rollout 是否成功（episode 層級，不考慮偵測時間）。

資料：outputs/results.csv 列出的每個 episode，
    特徵 outputs/<suite>/chunk_*_features/<key>.npz
    標籤 outputs/<suite>/chunk_*_bddl/<key>.npz（label: True = 成功）
每個 episode 的特徵先沿時間（chunk 維度）取平均，避免模型直接利用 episode 長度
（成功會提早結束、失敗會跑滿步數）。
但平均後的特徵仍帶有長度資訊（短的成功 episode 中「已完成」畫面佔比較高），
可用 --max_chunks k 只取前 k 個 chunk 再平均；k 小於最短 episode 時長度完全不可見。

模型：標準化 -> MLP（一層隱藏層），5-fold 分層交叉驗證（依 suite x 標籤分層）。

用法：
    python train_failure_detector.py                          # lm_last 第 9 層
    python train_failure_detector.py --feature vision_mean --layer 13
    python train_failure_detector.py --feature lm_mean --sweep  # 掃過每一層
    python train_failure_detector.py --feature lm_mean --sweep --max_chunks 6  # 只用前 6 個 chunk
"""

import argparse
import csv
import glob
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).parent / "outputs"


def load_dataset(max_chunks=None):
    """回傳 dict：每種特徵 (N, L, D) 的時間平均、label (N,)、n_chunks (N,)、suite (N,)。

    max_chunks 不為 None 時只平均前 max_chunks 個 chunk；n_chunks 仍是完整 episode 長度。
    """
    cache = ROOT / ("episode_mean_features.npz" if max_chunks is None else f"episode_mean_features_first{max_chunks}.npz")
    if cache.exists():
        d = np.load(cache)
        return {k: d[k] for k in d.files}
    rows = list(csv.DictReader(open(ROOT / "results.csv")))
    feat_files = {Path(f).stem: f for f in glob.glob(str(ROOT / "*/chunk_*_features/*.npz"))}
    label_files = {Path(f).stem: f for f in glob.glob(str(ROOT / "*/chunk_*_bddl/*.npz"))}
    out = {"lm_last": [], "lm_mean": [], "vision_mean": [], "label": [], "n_chunks": [], "suite": [], "category": []}
    for r in rows:
        key = f"{r['suite']}_{r['task_id']}_ep{r['episode']}"
        f = np.load(feat_files[key])
        t = slice(max_chunks)
        out["lm_last"].append(f["lm_last"][t].astype(np.float32).mean(0))
        out["lm_mean"].append(f["lm_mean"][t].astype(np.float32).mean(0))
        # 只取前兩個相機，第 3 個是空白相機
        out["vision_mean"].append(f["vision_mean"][t, :2].astype(np.float32).mean(0).reshape(-1, f["vision_mean"].shape[-1]))
        out["label"].append(bool(np.load(label_files[key])["label"][0]))
        out["n_chunks"].append(len(f["step"]))
        out["suite"].append(r["suite"])
        out["category"].append(r["category"])
    out = {k: np.array(v) for k, v in out.items()}
    np.savez(cache, **out)
    return out


def auroc(y, s):
    """Mann-Whitney U 形式的 AUROC（平手算 0.5）。"""
    y = np.asarray(y, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def stratified_folds(strata, k, seed):
    rng = np.random.default_rng(seed)
    fold = np.empty(len(strata), int)
    for s in np.unique(strata):
        idx = rng.permutation(np.where(strata == s)[0])
        fold[idx] = np.arange(len(idx)) % k
    return fold


def train_mlp(x_tr, y_tr, x_te, args, device, w_tr=None, logits=False):
    """w_tr 為每個樣本的權重（已含類別平衡）；None 時改用 pos_weight 平衡類別。
    logits=True 回傳 logit（不會像 sigmoid 在 float32 下飽和成 1.0，適合取分位數當門檻）。"""
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-6
    x_tr = torch.tensor((x_tr - mu) / sd, device=device)
    x_te = torch.tensor((x_te - mu) / sd, device=device)
    y_tr = torch.tensor(y_tr, dtype=torch.float32, device=device)
    model = nn.Sequential(
        nn.Dropout(args.input_dropout),
        nn.Linear(x_tr.shape[1], args.hidden),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(args.hidden, 1),
    ).to(device)
    if w_tr is None:
        pos_weight = (1 - y_tr.mean()) / y_tr.mean()  # 成功樣本較少，加權平衡
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        w_tr = torch.tensor(w_tr, dtype=torch.float32, device=device)
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for _ in range(args.epochs):
        model.train()
        perm = torch.randperm(len(x_tr), device=device)
        for i in range(0, len(perm), args.batch_size):
            b = perm[i : i + args.batch_size]
            opt.zero_grad()
            loss = loss_fn(model(x_tr[b]).squeeze(1), y_tr[b])
            if w_tr is not None:
                loss = (loss * w_tr[b]).mean()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        out = model(x_te).squeeze(1)
        return (out if logits else torch.sigmoid(out)).cpu().numpy()


def cross_validate(x, data, args, device):
    """回傳 (repeats, N)：每次重複交叉驗證的 out-of-fold 機率。"""
    y = data["label"]
    strata = np.char.add(data["suite"].astype(str), y.astype(str))
    prob = np.zeros((args.repeats, len(y)))
    for seed in range(args.repeats):
        torch.manual_seed(seed)
        fold = stratified_folds(strata, args.folds, seed)
        for k in range(args.folds):
            tr, te = fold != k, fold == k
            prob[seed, te] = train_mlp(x[tr], y[tr], x[te], args, device)
    return prob


def report(name, y, prob, data):
    pred = prob > 0.5
    tpr, tnr = (pred & y).sum() / y.sum(), (~pred & ~y).sum() / (~y).sum()
    print(f"{name}: AUROC={auroc(y, prob):.3f}  acc={(pred == y).mean():.3f}  "
          f"balanced_acc={(tpr + tnr) / 2:.3f}  (成功召回={tpr:.3f}, 失敗召回={tnr:.3f})")
    for s in np.unique(data["suite"]):
        m = data["suite"] == s
        print(f"    {s:<15} n={m.sum():<4} 成功={y[m].sum():<3} AUROC={auroc(y[m], prob[m]):.3f}  acc={(pred[m] == y[m]).mean():.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feature", default="lm_last", choices=["lm_last", "lm_mean", "vision_mean"])
    p.add_argument("--layer", type=int, default=9, help="vision_mean 的層編號為 cam*27+layer（0~53）")
    p.add_argument("--sweep", action="store_true", help="逐層跑交叉驗證，只印 AUROC")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--input_dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3, help="以不同切分重複交叉驗證，取平均機率")
    p.add_argument("--max_chunks", type=int, default=None, help="只用每個 episode 前 k 個 chunk 的特徵")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_dataset(args.max_chunks)
    y = data["label"]
    print(f"episodes={len(y)}  成功={y.sum()}  失敗={(~y).sum()}")
    if args.max_chunks is not None:
        short = data["n_chunks"] <= args.max_chunks
        print(f"只用前 {args.max_chunks} 個 chunk；長度 <= {args.max_chunks} 的 episode 有 {short.sum()} 個"
              f"（其中成功 {(short & y).sum()}），這些 episode 的長度仍可能被看出")

    # 對照：只看 episode 長度（chunk 數越少越可能成功）
    score = -data["n_chunks"].astype(float)
    print(f"[baseline] 只用 episode 長度: AUROC={auroc(y, score):.3f}")
    for s in np.unique(data["suite"]):
        m = data["suite"] == s
        print(f"    {s:<15} AUROC={auroc(y[m], score[m]):.3f}")
    # 各 suite 最大步數不同，suite 內標準化後才是公平的長度對照
    z = score.copy()
    for s in np.unique(data["suite"]):
        m = data["suite"] == s
        z[m] = (z[m] - z[m].mean()) / z[m].std()
    print(f"[baseline] suite 內標準化的 episode 長度: AUROC={auroc(y, z):.3f}")
    print()

    feats = data[args.feature]  # (N, L, D)
    if args.sweep:
        results = []
        for layer in range(feats.shape[1]):
            prob = cross_validate(feats[:, layer], data, args, device)
            per_repeat = [auroc(y, p) for p in prob]
            results.append((auroc(y, prob.mean(0)), layer))
            print(f"{args.feature} layer {layer:2d}: AUROC={results[-1][0]:.3f}  "
                  f"(每次重複 {np.mean(per_repeat):.3f} ± {np.std(per_repeat):.3f})")
        print("前 5 名:", "  ".join(f"L{l}={a:.3f}" for a, l in sorted(results, reverse=True)[:5]))
        return
    prob = cross_validate(feats[:, args.layer], data, args, device).mean(0)
    report(f"[MLP] {args.feature} layer {args.layer}", y, prob, data)


if __name__ == "__main__":
    main()
