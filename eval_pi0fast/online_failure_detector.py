"""線上失敗偵測：每個 chunk 依「到目前為止」的特徵給失敗分數，評估多早能發現失敗。

流程（每個 outer fold）：
  1. 分數（--score）：
       prefix  時間 t 的輸入是前 t+1 個 chunk 特徵的平均（嚴格因果），MLP 輸出失敗 logit s_t。
               訓練時每個 episode 的所有前綴都是樣本，每個 episode 總權重相同，再做類別平衡。
       cumsum  累加分數（同 SAFE-MLP）：s_t = Σ_{τ<=t} sigmoid(g(e_τ))，e_τ 為第 τ 個 chunk 的特徵，g 為 MLP。
               損失：失敗 episode Σ_t (t - s_t)、成功 episode Σ_t s_t，以類別頻率倒數加權；成功 episode 的分數
               也會隨時間上升，適合搭配 --threshold functional。
  2. 門檻：在訓練折內做 inner CV，取得訓練折中成功 episode 的 out-of-fold 分數曲線，
     再以 conformal 方法校準，使成功 episode 在任何時刻被誤報的比例 <= alpha：
       --threshold const       固定門檻 τ：「每個成功 episode 的最大分數」的 conformal 分位數
       --threshold functional  隨時間變化的門檻 τ(t) = μ(t) + h·m(t)（functional CP，同 SAFE）：
                               成功曲線分成兩份，一份估計逐時間的平均 μ(t) 與調變函數 m(t)（--fcp_modulation），
                               另一份以 max_t (s_t - μ(t)) / m(t) 為不一致分數，取其 conformal 分位數 h；
                               預設同 SAFE（tfunc / extend / 0.3）
  3. 偵測：測試折中第一次 s_t > τ(t) 的 chunk 即為偵測時間 T_det。

偵測期限（--deadline）：失敗 episode 都跑到 timeout，成功 episode 通常較早結束，只要等得夠久，
「跑得比較久」本身就能偵測失敗。min / median 時只計入同一基礎任務成功 episode 長度的最小值 / 中位數之前的警報
（同 SAFE 的 "by earliest stop"），門檻也只用這段分數校準。報表另附「長度基準」：分數 s_t = t、同樣的折與門檻，
模型要明顯勝過它，才代表不是靠長度偵測。

交叉驗證的切分方式（outer 與 inner 相同）：
  --split episode  依 suite x 成功/失敗分層，隨機切分 episode；同一個基礎任務會同時出現在訓練與測試（看過的任務）
  --split task     以基礎任務為單位切分（每個 suite 的基礎任務平均分到各折），測試的任務訓練時都沒看過（未見任務）
  LIBERO-plus 的任務 = 原始 LIBERO 任務（每個 suite 10 個）+ 一種擾動，基礎任務由任務名稱去掉擾動後綴得到。

失敗 episode 都跑到 timeout，沒有失敗起點的標註，所以偵測時間以「已看過的 chunk 數」
及「佔 timeout 的比例」表示；早期偵測分數 = 失敗 episode 平均剩餘時間比例（未偵測算 0）。

預設分數為 cumsum（門檻預設 functional）、輸入為 sub_first。
原本的預設值（lm_mean、第 12 層、prefix、const、conformal、deadline=min）取自 lm_mean / lm_last 全部層 x 兩種分數 x
各種門檻的 sweep（以 deadline=min、α=0.1 的早期偵測分數排序，前幾名再以 3 個 repeat 確認）；
第 7、10、11、12 層差距在 0.01 內，屬於誤差範圍（該 sweep 為 --input raw）。

不依賴初始圖像 / 語言（--input，預設 sub_first）：每個 chunk 的特徵都含有初始場景與語言指令，
只用第一個 chunk（--input first）就有 episode AUROC 0.83（--split task 為 0.67），raw 因此常在第 1 個 chunk 就警報，
等於在猜「這個任務 / 擾動容易失敗」而非偵測執行出錯。sub_first 用 e_t - e_0，t=1 的分數對所有 episode 相同，
偵測只能來自 episode 開始後的變化；delta 用 e_t - e_{t-1}；--window 只平均最近幾個 chunk；
--skip k 略過前 k 個 chunk（raw 下的特徵仍含靜態資訊，只是延後）。

用法：
    python online_failure_detector.py --feature lm_mean --layer 12
    python online_failure_detector.py --feature lm_mean --sweep --repeats 1
    python online_failure_detector.py --feature lm_mean --layer 10 --threshold functional
    python online_failure_detector.py --feature lm_mean --layer 10 --split task
    python online_failure_detector.py --feature lm_mean --layer 10 --score cumsum --threshold functional
    python online_failure_detector.py --feature lm_mean --sweep --repeats 1 --score cumsum --threshold functional
    python online_failure_detector.py --feature lm_mean --layer 10 --deadline min
    python online_failure_detector.py --input first      # 診斷：只看第一個 chunk 能偵測多少
    python online_failure_detector.py --input raw        # 舊行為：直接用原始特徵
"""

import argparse
import csv
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# 評測輸出資料夾；可用環境變數 OUTPUTS_DIR 指定（例如重跑評估後的 outputs_act）
ROOT = Path(os.environ.get("OUTPUTS_DIR", Path(__file__).parent / "outputs"))

# LIBERO-plus 任務名稱中的擾動後綴，例如 _table_23、_light_33、_language_42_view_...、_add_19、_moved_level1_sample1
PERTURBATION_SUFFIX = re.compile(r"_(table_\d+|tb_\d+|view_\d+_|language_\d+|light_\d+|add_\d+|moved_|level\d+_sample\d+).*$")


def base_task(task_name):
    """去掉擾動後綴，得到原始 LIBERO 任務名稱（每個 suite 10 個）。"""
    return PERTURBATION_SUFFIX.sub("", task_name)


def load_dataset():
    """依 outputs/results.csv 的順序，回傳 label (N,)（True = 成功）、n_chunks (N,)、suite (N,)、category (N,)、
    base_task (N,)（"<suite>/<原始任務名稱>"）。"""
    rows = list(csv.DictReader(open(ROOT / "results.csv")))
    feat_files = {Path(f).stem: f for f in glob.glob(str(ROOT / "*/chunk_*_features/*.npz"))}
    label_files = {Path(f).stem: f for f in glob.glob(str(ROOT / "*/chunk_*_bddl/*.npz"))}
    out = {"label": [], "n_chunks": [], "suite": [], "category": [], "base_task": []}
    for r in rows:
        key = f"{r['suite']}_{r['task_id']}_ep{r['episode']}"
        out["label"].append(bool(np.load(label_files[key])["label"][0]))
        out["n_chunks"].append(len(np.load(feat_files[key])["step"]))
        out["suite"].append(r["suite"])
        out["category"].append(r["category"])
        out["base_task"].append(f"{r['suite']}/{base_task(r['task_name'])}")
    return {k: np.array(v) for k, v in out.items()}


def auroc(y, s):
    """Mann-Whitney U 形式的 AUROC（平手算 0.5）。"""
    y = np.asarray(y, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def make_folds(idx, data, k, seed, split):
    """對 idx 中的 episode 切出 k 折（回傳與 idx 等長的折編號）。
    task = 同一個基礎任務放在同一折，每個 suite 內的基礎任務平均分到各折；episode = 依 suite x 成功/失敗分層隨機切分。"""
    rng = np.random.default_rng(seed)
    fold = np.empty(len(idx), int)
    suite = data["suite"][idx]
    if split == "task":
        groups = data["base_task"][idx]
        for s in np.unique(suite):
            for i, g in enumerate(rng.permutation(np.unique(groups[suite == s]))):
                fold[groups == g] = i % k
        return fold
    strata = np.char.add(suite.astype(str), data["label"][idx].astype(str))
    for s in np.unique(strata):
        j = rng.permutation(np.where(strata == s)[0])
        fold[j] = np.arange(len(j)) % k
    return fold


def train_mlp(x_tr, y_tr, x_te, w_tr, args, device):
    """標準化 -> MLP，以每個樣本的權重 w_tr（已含類別平衡）訓練，回傳 x_te 的 logit
    （不會像 sigmoid 在 float32 下飽和成 1.0，適合取分位數當門檻）。"""
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-6
    x_tr = torch.tensor((x_tr - mu) / sd, device=device)
    x_te = torch.tensor((x_te - mu) / sd, device=device)
    y_tr = torch.tensor(y_tr, dtype=torch.float32, device=device)
    w_tr = torch.tensor(w_tr, dtype=torch.float32, device=device)
    model = nn.Sequential(
        nn.Dropout(args.input_dropout),
        nn.Linear(x_tr.shape[1], args.hidden),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(args.hidden, 1),
    ).to(device)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for _ in range(args.epochs):
        model.train()
        perm = torch.randperm(len(x_tr), device=device)
        for i in range(0, len(perm), args.batch_size):
            b = perm[i : i + args.batch_size]
            opt.zero_grad()
            loss = (loss_fn(model(x_tr[b]).squeeze(1), y_tr[b]) * w_tr[b]).mean()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return model(x_te).squeeze(1).cpu().numpy()


def load_chunk_features(feature, data):
    """回傳 (L, N, Tmax, D) memmap，時間超過 n_chunks 的部分為 0。順序同 load_dataset。
    一律存成 float32：特徵檔可能混有舊的 float16 與新的 float32，float16 快取會讓深層數值再次溢位成 inf。
    舊的 float16 檔案本身在深層可能已溢位，見 nonfinite_layers。"""
    cache = ROOT / f"chunk_features_{feature}.npy"
    if not cache.exists():
        rows = list(csv.DictReader(open(ROOT / "results.csv")))
        feat_files = {Path(f).stem: f for f in glob.glob(str(ROOT / "*/chunk_*_features/*.npz"))}
        first = np.load(feat_files[f"{rows[0]['suite']}_{rows[0]['task_id']}_ep{rows[0]['episode']}"])[feature]
        # 每個程序用自己的暫存檔，同時執行多個程序時不會互相覆寫；寫完才改名，中斷時不會留下壞掉的快取
        tmp = cache.with_suffix(f".tmp{os.getpid()}.npy")
        out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32,
                                        shape=(first.shape[1], len(rows), int(data["n_chunks"].max()), first.shape[-1]))
        for i, r in enumerate(rows):
            f = np.load(feat_files[f"{r['suite']}_{r['task_id']}_ep{r['episode']}"])[feature]
            out[:, i, : len(f)] = f.transpose(1, 0, 2)
        out.flush()
        del out
        os.replace(tmp, cache)
    return np.load(cache, mmap_mode="r")


def nonfinite_layers(F):
    """含 inf / NaN 的層（舊的 float16 特徵檔在深層會溢位）。"""
    bad = [l for l in range(F.shape[0]) if not np.isfinite(F[l]).all()]
    if bad:
        print(f"警告：第 {bad} 層特徵含 inf / NaN（舊的特徵檔以 float16 儲存而溢位），這些層無法使用；"
              f"要取得這些層需以 float32 的 pi0fast_features.py 重新評估，並刪除 chunk_features_*.npy 快取。", flush=True)
    return bad


def transform_features(f, n, mode):
    """f (N, Tmax, D)、n (N,) -> 去除 episode 靜態內容後的逐 chunk 特徵（超出長度為 0）。
      raw       原始特徵
      sub_first e_t - e_0：減掉該 episode 第一個 chunk 的特徵，去掉初始圖像 / 語言指令等不隨時間變的成分
      delta     e_t - e_{t-1}：只看相鄰 chunk 的變化（e_0 的位置為 0）
      first     診斷用：每個時間點都只給 e_0，衡量「只看一開始」能偵測到多少
    """
    mask = (np.arange(f.shape[1])[None] < n[:, None])[..., None]
    if mode == "sub_first":
        f = f - f[:, :1]
    elif mode == "delta":
        f = np.concatenate([np.zeros_like(f[:, :1]), np.diff(f, axis=1)], 1)
    elif mode == "first":
        f = np.broadcast_to(f[:, :1], f.shape)
    return np.where(mask, f, 0).astype(np.float32)


def score_inputs(f, args):
    """f (N, Tmax, D) 已轉換（transform_features）的逐 chunk 特徵 -> 分數模型的輸入 (N, Tmax, D)：
    prefix 取每個時間點的前綴平均（--window > 0 時只平均最近 window 個 chunk），cumsum 直接用逐 chunk 特徵。"""
    if args.score == "cumsum":
        return f
    c = np.cumsum(f, axis=1, dtype=np.float64)
    cnt = np.arange(1, f.shape[1] + 1, dtype=np.float64)
    if args.window > 0:
        c[:, args.window:] -= c[:, : -args.window].copy()
        cnt = np.minimum(cnt, args.window)
    return (c / cnt[None, :, None]).astype(np.float32)


def effective_skip(args):
    """sub_first / delta 時第 0 個 chunk 的輸入對所有 episode 都是 0、不含資訊，至少略過它：
    否則所有成功曲線在 t=0 完全相同，functional 門檻的 m(0)≈0，inner / outer 模型輸出的些微差異就會在 t=0 誤報。"""
    return max(args.skip, 1) if args.input in ("sub_first", "delta") else args.skip


def skip_chunks(F, data, k):
    """去掉每個 episode 的前 k 個 chunk（F 為 (L, N, Tmax, D) 或 (N, Tmax, D)），回傳平移後的特徵與 data；
    之後的分數 / 門檻再以 pad_skipped 補回原本的時間軸。"""
    if k == 0:
        return F, data
    if (data["n_chunks"] <= k).any():
        raise ValueError(f"--skip {k} 不能 >= 最短 episode 長度 {data['n_chunks'].min()}")
    d = dict(data, n_chunks=data["n_chunks"] - k, deadline=np.maximum(data["deadline"] - k, 0))
    return F[..., k:, :], d


def pad_skipped(scores, thr, k):
    """把平移後的 scores (R, N, T) / thr (R, A, N, T) 前面補 k 個時間點：分數 nan、門檻 inf（不會警報）。"""
    if k == 0:
        return scores, thr
    pad = lambda x, v: np.concatenate([np.full(x.shape[:-1] + (k,), v), x], -1)
    return pad(scores, np.nan), pad(thr, np.inf)


def fit_predict_cumsum(X, n, fail, tr, te, args, device):
    """SAFE-MLP：用 tr 訓練 g，回傳 te 的逐 chunk 累加分數 s_t = Σ_{τ<=t} sigmoid(g(e_τ)) (len(te), Tmax)，超出長度為 nan。

    損失（同 SAFE）：失敗 episode Σ_t (t - s_t)、成功 episode Σ_t s_t，每個 episode 以類別頻率倒數加權；
    L2 正則化由 AdamW 的 weight_decay 提供；每個 batch 為 args.batch_size 個 episode。
    --cumsum_bce λ > 0 時每個 episode 另加 λ·Σ_t BCE(g(e_t), 失敗)：原損失對 logit 的梯度含 p(1-p)，
    sigmoid 飽和（p=1）後幾乎為 0，被推到 p=1 的成功 episode 拉不回來，分數與門檻都貼在上限 t 附近。
    """
    T = X.shape[1]
    mask = np.arange(T)[None] < n[:, None]
    valid = X[tr][mask[tr]]
    mu, sd = valid.mean(0), valid.std(0) + 1e-6
    x_tr = torch.tensor((X[tr] - mu) / sd, device=device)
    x_te = torch.tensor((X[te] - mu) / sd, device=device)
    m_tr = torch.tensor(mask[tr], dtype=torch.float32, device=device)
    m_te = torch.tensor(mask[te], dtype=torch.float32, device=device)
    y_tr = torch.tensor(fail[tr], dtype=torch.float32, device=device)
    w_tr = torch.where(y_tr == 1, 1 / y_tr.mean(), 1 / (1 - y_tr.mean()))
    w_tr /= w_tr.mean()
    t = torch.arange(1, T + 1, dtype=torch.float32, device=device)
    model = nn.Sequential(
        nn.Dropout(args.input_dropout),
        nn.Linear(X.shape[-1], args.hidden),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(args.hidden, 1),
    ).to(device)

    def score(x, m, logits=False):
        z = model(x).squeeze(-1)
        s = torch.cumsum(torch.sigmoid(z) * m, 1)
        return (s, z) if logits else s

    bce = nn.BCEWithLogitsLoss(reduction="none")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for _ in range(args.epochs):
        model.train()
        perm = torch.randperm(len(tr), device=device)
        for i in range(0, len(perm), args.batch_size):
            b = perm[i : i + args.batch_size]
            m = m_tr[b]
            s, z = score(x_tr[b], m, logits=True)
            loss_ep = torch.where(y_tr[b] == 1, ((t - s) * m).sum(1), (s * m).sum(1))
            if args.cumsum_bce > 0:  # 逐 chunk BCE：sigmoid 飽和時仍有梯度，把被推到 p=1 的成功 chunk 拉回來
                loss_ep = loss_ep + args.cumsum_bce * (bce(z, y_tr[b, None].expand_as(z)) * m).sum(1)
            opt.zero_grad()
            (loss_ep * w_tr[b]).mean().backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        s = score(x_te, m_te).cpu().numpy()
    return np.where(mask[te], s, np.nan)


def fit_predict(P, n, fail, tr, te, args, device):
    """用 tr 中 episode 的所有前綴訓練，回傳 te 的逐 chunk 失敗分數 (len(te), Tmax)，超出長度為 nan。"""
    if args.score == "cumsum":
        return fit_predict_cumsum(P, n, fail, tr, te, args, device)
    x_tr = np.concatenate([P[i, : n[i]] for i in tr])
    y_tr = np.repeat(fail[tr], n[tr]).astype(np.float32)
    w = np.repeat(1.0 / n[tr], n[tr])  # 每個 episode 總權重相同
    w[y_tr == 1] *= w[y_tr == 0].sum() / w[y_tr == 1].sum()  # 類別平衡
    w /= w.mean()
    s = train_mlp(x_tr, y_tr, np.concatenate([P[i, : n[i]] for i in te]), w, args, device)
    out = np.full((len(te), P.shape[1]), np.nan)
    for j, (i, o) in enumerate(zip(te, np.cumsum(n[te]) - n[te])):
        out[j, : n[i]] = s[o : o + n[i]]
    return out


def conformal_quantile(r, alpha):
    """使新樣本的不一致分數超過回傳值的機率 <= alpha；樣本太少時為 inf（永不警報）。"""
    m = np.sort(r[~np.isnan(r)])  # 期限為 0 的 episode 沒有分數
    k =int(np.ceil((len(m) + 1) * (1 - alpha)))
    return np.inf if k > len(m) else m[k - 1]


def const_threshold(curves, alpha):
    """curves (M, Tmax) 成功 episode 的分數曲線（超出長度為 nan）-> 固定門檻 (Tmax,)。"""
    return np.full(curves.shape[1], conformal_quantile(np.nanmax(curves, 1), alpha))


def functional_threshold(curves, alpha, seed, args):
    """curves (M, Tmax) 成功 episode 的分數曲線 -> 隨時間變化的門檻 μ(t) + h·m(t) (Tmax,)。

    比例 args.fcp_fit_frac 的成功 episode 估計平均 μ(t) 與調變函數 m(t)，其餘以 conformal 分位數估計 h，
    conformal 保證才成立。
      m(t)：std   = 逐時間標準差 σ(t)
            tfunc = 去掉最極端（最大偏差排名超過 conformal 分位數）的曲線後，逐時間的最大 |s_t - μ(t)|（同 SAFE）
    成功 episode 會提早結束：
      freeze = 仍在進行的 episode 少於 args.fcp_min_count 的時間點沿用最後一個可靠的 μ、m
      extend = 先把每條曲線以最後一個分數補到 Tmax（同 SAFE），μ、m 由所有成功 episode 估計
    """
    if args.fcp_align == "extend":
        last = curves[np.arange(len(curves)), (~np.isnan(curves)).sum(1) - 1]
        curves = np.where(np.isnan(curves), last[:, None], curves)
    idx = np.random.default_rng(seed).permutation(len(curves))
    n_fit = min(max(int(len(idx) * args.fcp_fit_frac), 1), len(idx) - 1)
    fit, cal = curves[idx[:n_fit]], curves[idx[n_fit:]]
    valid = (~np.isnan(fit)).sum(0) >= min(args.fcp_min_count, n_fit)  # 隨時間遞減，所以是開頭連續的一段
    last = int(np.nonzero(valid)[0].max()) if valid.any() else 0
    mu = np.full(curves.shape[1], np.nan)
    md = np.full(curves.shape[1], np.nan)
    with np.errstate(all="ignore"):
        mu[: last + 1] = np.nanmean(fit[:, : last + 1], 0)
        if args.fcp_modulation == "tfunc":
            dev = np.abs(fit[:, : last + 1] - mu[: last + 1])
            gamma = conformal_quantile(np.nanmax(dev, 1), alpha)  # 樣本太少時為 inf，即保留全部曲線
            md[: last + 1] = np.nanmax(dev[np.nanmax(dev, 1) <= gamma], 0)
        else:
            md[: last + 1] = np.nanstd(fit[:, : last + 1], 0)
    md[: last + 1] = np.where(np.isnan(md[: last + 1]), np.nanmean(md), md[: last + 1])
    mu[last + 1 :], md[last + 1 :] = mu[last], md[last]
    md = np.maximum(md, 1e-3 * np.nanmean(md) + 1e-6)  # 避免 m 過小讓分數爆掉
    h = conformal_quantile(np.nanmax((cal - mu) / md, 1), alpha)
    return mu + h * md


def detection_deadline(data, mode):
    """每個 episode 的偵測期限 (N,)：只計入前 deadline 個 chunk 內的警報。
    none = 整個 episode；min / median = 同一基礎任務成功 episode 長度的最小值 / 中位數，
    沒有成功 episode 的基礎任務改用同 suite 的值。期限以全部 episode 計算，屬於評估協定（同 SAFE）。"""
    y, n = data["label"], data["n_chunks"]
    if mode == "none":
        return n.copy()
    stat = np.min if mode == "min" else np.median
    d = np.zeros(len(n), int)
    for key in ("suite", "base_task"):  # 先填 suite 的值，有成功 episode 的基礎任務再覆蓋
        for g in np.unique(data[key]):
            m = data[key] == g
            if (m & y).any():
                d[m] = int(np.floor(stat(n[m & y])))
    return np.minimum(d, n)


def within_deadline(s, d):
    """s (M, Tmax) 分數曲線、d (M,) 期限 -> 期限之後設為 nan。"""
    return np.where(np.arange(s.shape[1])[None] < d[:, None], s, np.nan)


def episode_auroc(scores, data):
    """episode 層級 AUROC：期限內最大失敗分數（各 repeat 平均）。"""
    return float(np.mean([auroc(~data["label"], np.nanmax(within_deadline(s, data["deadline"]), 1)) for s in scores]))


def length_baseline(data, args):
    """只看時間長度的基準：分數 s_t = t，門檻由訓練折成功 episode（截到期限）的長度校準，
    與模型使用相同的折；回傳格式同 run_online。"""
    y, n, d = data["label"], data["n_chunks"], data["deadline"]
    s = within_deadline(np.tile(np.arange(1, n.max() + 1, dtype=float), (len(y), 1)), n)
    thr = np.zeros((args.repeats, len(args.alphas), len(y), s.shape[1]))
    for seed in range(args.repeats):
        fold = make_folds(np.arange(len(y)), data, args.folds, seed, args.split)
        for k in range(args.folds):
            cal = np.where((fold != k) & y)[0]
            for a, alpha in enumerate(args.alphas):
                thr[seed, a, fold == k] = const_threshold(within_deadline(s[cal], d[cal]), alpha)
    return np.repeat(s[None], args.repeats, 0), thr


def run_online(P, data, args, device):
    """回傳 scores (repeats, N, Tmax) 與 thresholds (repeats, A, N, Tmax)：每個 episode 在其測試折的逐時間門檻。"""
    y, n = data["label"], data["n_chunks"]
    fail = ~y
    scores = np.full((args.repeats, len(y), P.shape[1]), np.nan)
    thr = np.zeros((args.repeats, len(args.alphas), len(y), P.shape[1]))
    for seed in range(args.repeats):
        torch.manual_seed(seed)
        fold = make_folds(np.arange(len(y)), data, args.folds, seed, args.split)
        for k in range(args.folds):
            tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
            inner = make_folds(tr, data, args.calib_folds, seed + 1000, args.split)
            curves, cal = [], []
            for j in range(args.calib_folds):
                itr, ite = tr[inner != j], tr[inner == j]
                ite = ite[y[ite]]  # 只需成功 episode 的分數
                if len(ite) == 0:
                    continue
                curves.append(fit_predict(P, n, fail, itr, ite, args, device))
                cal.append(ite)
            curves = within_deadline(np.concatenate(curves), data["deadline"][np.concatenate(cal)])
            curves = curves[~np.isnan(curves).all(1)]  # --skip 後期限為 0 的 episode
            for a, alpha in enumerate(args.alphas):
                if args.threshold == "functional":
                    tau = functional_threshold(curves, alpha, seed * 100 + k, args)
                else:
                    tau = const_threshold(curves, alpha)
                thr[seed, a, te] = tau
            scores[seed, te] = fit_predict(P, n, fail, tr, te, args, device)
    return scores, thr


def run_layer(f, data, args, device):
    """某一層的特徵 f (N, Tmax, D) -> 原本時間軸上的 scores、thresholds（含 --input、--skip、--window）。
    先以完整 episode 轉換特徵（sub_first 的參考仍是 e_0），再略過前 k 個 chunk。"""
    k = effective_skip(args)
    f, d = skip_chunks(transform_features(np.asarray(f, dtype=np.float32), data["n_chunks"], args.input), data, k)
    scores, thr = run_online(score_inputs(f, args), d, args, device)
    return pad_skipped(scores, thr, k)


def is_alarm(s, tau):
    """s > τ 才警報；nan 不警報。cumsum 的 sigmoid 常飽和成 1.0，使 s_t 與門檻都剛好等於 t，
    兩者相等時浮點誤差會讓 s 略大於 τ 而誤報（conformal 定義下相等不算超過），所以要求超過一個極小的容差。"""
    return np.nan_to_num(s, nan=-np.inf) > tau + 1e-4 * np.maximum(1, np.abs(tau))


def detection_metrics(s, tau, data, mask=None):
    """s (N, Tmax) 失敗分數、tau (N, Tmax) 逐時間門檻 -> 指標 dict；只計入偵測期限內的警報。"""
    y, n, d = data["label"], data["n_chunks"], data["deadline"]
    if mask is not None:
        s, tau, y, n, d = s[mask], tau[mask], y[mask], n[mask], d[mask]
    alarm = is_alarm(within_deadline(s, d), tau)
    detected = alarm.any(1)
    t_det = alarm.argmax(1) + 1  # 偵測時已看過的 chunk 數
    frac = t_det / n  # 失敗 episode 的 n = timeout
    f = ~y
    df = detected & f
    return {
        "TPR": detected[f].mean(),
        "FPR": detected[y].mean(),
        "BA": (detected[f].mean() + (1 - detected[y].mean())) / 2,
        "T_det_mean": t_det[df].mean() if df.any() else np.nan,
        "T_det_median": np.median(t_det[df]) if df.any() else np.nan,
        "frac_median": np.median(frac[df]) if df.any() else np.nan,
        "frac_mean": frac[df].mean() if df.any() else np.nan,
        "early_score": np.where(df, 1 - frac, 0)[f].mean(),
        "cum": [(df & (frac <= q))[f].mean() for q in np.linspace(0.1, 1, 10)],
    }


def mean_metrics(scores, thr, a, data, mask=None):
    ms = [detection_metrics(scores[r], thr[r, a], data, mask) for r in range(len(scores))]
    return {k: np.nanmean([m[k] for m in ms], axis=0) for k in ms[0]}


def time_points(n):
    return list(range(1, n.min() + 1)) + [t for t in (10, 15, 20, 25) if t > n.min()]


def auroc_at(scores, y, n, t):
    """第 t 個 chunk 時，仍在進行的 episode 的 AUROC（各 repeat 平均）。"""
    if np.isnan(scores[:, n >= t, t - 1]).all():  # --skip 略過的時間點
        return float("nan")
    return float(np.mean([auroc(~y[n >= t], s[n >= t, t - 1]) for s in scores]))


def report(name, scores, thr, data, args, base):
    """base = length_baseline 的 (scores, thr)。"""
    y, n = data["label"], data["n_chunks"]
    print(f"[{name}]  分數={args.score}  門檻={args.threshold}  切分={args.split}  偵測期限={args.deadline}  "
          f"輸入={args.input}  window={args.window}  skip={args.skip}  cumsum_bce={args.cumsum_bce}")
    print(f"  episode 層級 AUROC（期限內最大失敗分數）= {episode_auroc(scores, data):.3f}"
          f"（長度基準 {episode_auroc(base[0], data):.3f}）")
    # 每個時間點只看仍在進行的 episode；t <= 最短 episode 時所有 episode 都在
    line = [f"t={t}:{auroc_at(scores, y, n, t):.3f}" for t in time_points(n)]
    print("  逐時間 AUROC（仍在進行的 episode）:", "  ".join(line))
    print()
    for a, alpha in enumerate(args.alphas):
        m = mean_metrics(scores, thr, a, data)
        print(f"  alpha={alpha:.2f}: 失敗偵測率={m['TPR']:.3f}  成功誤報率={m['FPR']:.3f}  "
              f"T_det 中位數={m['T_det_median']:.1f} chunks  佔 timeout 中位數={m['frac_median']:.2f}（平均 {m['frac_mean']:.2f}）  "
              f"早期偵測分數={m['early_score']:.3f}")
        print("      累積偵測率 @ timeout 的", "  ".join(f"{q:.0%}:{c:.2f}" for q, c in zip(np.linspace(0.1, 1, 10), m["cum"])))
        mb = mean_metrics(*base, a, data)
        print(f"      長度基準: 失敗偵測率={mb['TPR']:.3f}  成功誤報率={mb['FPR']:.3f}  "
              f"T_det 中位數={mb['T_det_median']:.1f} chunks  早期偵測分數={mb['early_score']:.3f}")
        if args.threshold == "functional":
            tau = np.nanmean(thr[:, a], axis=(0, 1))
            print("      門檻 τ(t) 平均:", "  ".join(f"t={t}:{tau[t - 1]:.2f}" for t in time_points(n)))
    a = args.alphas.index(args.alpha)
    print(f"\n  各 suite（alpha={args.alpha:.2f}）:")
    for s in np.unique(data["suite"]):
        mask = data["suite"] == s
        m = mean_metrics(scores, thr, a, data, mask)
        print(f"    {s:<15} 成功={y[mask].sum():<3} 失敗={(~y[mask]).sum():<3} timeout={n[mask & ~y].max()} chunks  "
              f"偵測率={m['TPR']:.3f}  誤報率={m['FPR']:.3f}  T_det 中位數={m['T_det_median']:.1f}  "
              f"佔 timeout={m['frac_median']:.2f}  早期偵測分數={m['early_score']:.3f}")

    # 摘要表：balanced accuracy 與平均偵測時間（只算有偵測到的失敗 episode）
    hdr = "| {:<22} | {:>5} | {:>5} | {:>5} | {:>5} | {:>13} | {:>13} |".format(
        "設定", "TPR", "FPR", "BA", "early", "T_det 平均", "佔 timeout 平均")
    print(f"\n  摘要表（BA = (TPR + 1 - FPR) / 2）:\n  {hdr}\n  |{'|'.join('-' * w for w in (24, 7, 7, 7, 7, 15, 15))}|")
    rows = [(f"all α={alpha:.2f}", a, None, (scores, thr)) for a, alpha in enumerate(args.alphas)]
    rows += [(f"長度基準 α={alpha:.2f}", a, None, base) for a, alpha in enumerate(args.alphas)]
    rows += [(f"{s} α={args.alpha:.2f}", a, data["suite"] == s, (scores, thr)) for s in np.unique(data["suite"])]
    for label, ai, mask, (sc, th) in rows:
        m = mean_metrics(sc, th, ai, data, mask)
        print(f"  | {label:<22} | {m['TPR']:.3f} | {m['FPR']:.3f} | {m['BA']:.3f} | {m['early_score']:.3f} | "
              f"{m['T_det_mean']:>13.1f} | {m['frac_mean']:>13.2f} |")


def write_sheet(ws, header, rows, fmts, widths):
    """寫入一張表：Arial、藍底標題列、凍結標題、依欄設定數字格式；nan 寫成空白。"""
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    font, bold = Font(name="Arial", size=10), Font(name="Arial", size=10, bold=True)
    fill, line = PatternFill("solid", fgColor="D9E1F2"), Side(style="thin", color="BFBFBF")
    ws.append(header)
    for r in rows:
        ws.append([None if isinstance(v, float) and np.isnan(v) else v for v in r])
    for row in ws.iter_rows():
        for c in row:
            c.font = bold if c.row == 1 else font
            c.border = Border(bottom=line)
            if c.row == 1:
                c.fill, c.alignment = fill, Alignment(horizontal="center", vertical="center", wrap_text=True)
            elif fmts[c.column - 1]:
                c.number_format = fmts[c.column - 1]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A2"


def add_note(ws, text):
    from openpyxl.styles import Font

    ws.cell(ws.max_row + 2, 1, text).font = Font(name="Arial", size=9, italic=True)


def settings_rows(args, **extra):
    return [[k, v] for k, v in {**extra, **{k: (",".join(map(str, v)) if isinstance(v, list) else v)
                                            for k, v in vars(args).items() if k not in ("sweep", "xlsx")}}.items()]


def save_xlsx(path, name, scores, thr, data, args, base):
    """把摘要、累積偵測率、逐時間 AUROC 與設定寫成試算表；BA 以公式計算；base = length_baseline 的 (scores, thr)。"""
    from openpyxl import Workbook

    y, n = data["label"], data["n_chunks"]
    scopes = [("all", None, (scores, thr)), ("長度基準 (all)", None, base)]
    scopes += [(s, data["suite"] == s, (scores, thr)) for s in np.unique(data["suite"])]
    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    rows, cum_rows = [], []
    for a, alpha in enumerate(args.alphas):
        for scope, mask, (sc, th) in scopes:
            mk = np.ones(len(y), bool) if mask is None else mask
            m = mean_metrics(sc, th, a, data, mask)
            r = len(rows) + 2
            rows.append([scope, alpha, int(y[mk].sum()), int((~y[mk]).sum()), int(n[mk & ~y].max()),
                         m["TPR"], m["FPR"], f"=(F{r}+1-G{r})/2", m["early_score"],
                         m["T_det_mean"], m["T_det_median"], m["frac_mean"], m["frac_median"]])
            cum_rows.append([scope, alpha, *m["cum"]])
    write_sheet(ws, ["Scope", "α", "成功 episodes", "失敗 episodes", "Timeout (chunks)", "TPR（失敗偵測率）",
                     "FPR（成功誤報率）", "Balanced Acc.", "早期偵測分數", "平均 T_det (chunks)", "T_det 中位數 (chunks)",
                     "平均佔 timeout", "佔 timeout 中位數"],
                rows, [None, "0.00", "0", "0", "0", "0.000", "0.000", "0.000", "0.000", "0.0", "0.0", "0.00", "0.00"],
                [16, 6, 10, 10, 11, 12, 12, 11, 11, 12, 12, 11, 11])
    add_note(ws, "Balanced Acc. = (TPR + 1 − FPR) / 2；T_det 只計入有被偵測到的失敗 episode；"
                 f"偵測期限 = {args.deadline}；長度基準 = 只用時間 t 當分數；"
                 f"數值為 {args.repeats} 次重複 × {args.folds} 折的平均。")

    write_sheet(wb.create_sheet("Cumulative"), ["Scope", "α"] + [f"≤{q:.0%} timeout" for q in np.linspace(0.1, 1, 10)],
                cum_rows, [None, "0.00"] + ["0.000"] * 10, [16, 6] + [11] * 10)

    write_sheet(wb.create_sheet("AUROC by time"), ["t (chunks)", "仍在進行的 episodes", "AUROC"],
                [[t, int((n >= t).sum()), auroc_at(scores, y, n, t)] for t in time_points(n)]
                + [["episode 層級（期限內最大分數）", len(y), episode_auroc(scores, data)]],
                [None, "0", "0.000"], [24, 18, 10])

    write_sheet(wb.create_sheet("Settings"), ["參數", "值"], settings_rows(args, 模型=name), [None, None], [16, 28])
    wb.save(path)
    print(f"\n已寫出 {path}")


def layer_results(scores, thr, data, args):
    """單一層的所有指標：各 alpha 的整體指標、--alpha 下各 suite 指標、逐時間 AUROC。"""
    y, n = data["label"], data["n_chunks"]
    a = args.alphas.index(args.alpha)
    return {
        "auroc": episode_auroc(scores, data),
        "alpha": [mean_metrics(scores, thr, i, data) for i in range(len(args.alphas))],
        "suite": {s: mean_metrics(scores, thr, a, data, data["suite"] == s) for s in np.unique(data["suite"])},
        "time": [auroc_at(scores, y, n, t) for t in time_points(n)],
    }


def save_sweep_xlsx(path, feature, results, data, args, base):
    """逐層比較表：每個 alpha 一張（每列一層，最後一列為長度基準），另有各 suite、逐時間 AUROC、設定。
    base = 長度基準的 layer_results。"""
    from openpyxl import Workbook
    from openpyxl.formatting.rule import ColorScaleRule
    from openpyxl.utils import get_column_letter

    def color(ws, col, nrows, higher_better=True):
        lo, hi = ("F8696B", "63BE7B") if higher_better else ("63BE7B", "F8696B")
        ws.conditional_formatting.add(f"{col}2:{col}{nrows + 1}", ColorScaleRule(
            start_type="min", start_color=lo, mid_type="percentile", mid_value=50, mid_color="FFEB84",
            end_type="max", end_color=hi))

    wb = Workbook()
    wb.remove(wb.active)
    for i, alpha in enumerate(args.alphas):
        ws = wb.create_sheet(f"Layers α={alpha:g}")
        rows = []
        for layer, r in results + [("長度基準", base)]:
            m, k = r["alpha"][i], len(rows) + 2
            rows.append([layer, r["auroc"], m["TPR"], m["FPR"], f"=(C{k}+1-D{k})/2", m["early_score"],
                         m["T_det_mean"], m["T_det_median"], m["frac_mean"], m["frac_median"]])
        write_sheet(ws, ["Layer", "Episode AUROC", "TPR（失敗偵測率）", "FPR（成功誤報率）", "Balanced Acc.",
                         "早期偵測分數", "平均 T_det (chunks)", "T_det 中位數 (chunks)", "平均佔 timeout", "佔 timeout 中位數"],
                    rows, ["0", "0.000", "0.000", "0.000", "0.000", "0.000", "0.0", "0.0", "0.00", "0.00"],
                    [8, 11, 12, 12, 11, 11, 12, 12, 11, 11])
        for col, hb in [("B", True), ("C", True), ("D", False), ("E", True), ("F", True), ("G", False), ("I", False)]:
            color(ws, col, len(results), hb)
        add_note(ws, f"{feature}，α={alpha:g}，偵測期限={args.deadline}。Balanced Acc. = (TPR + 1 − FPR) / 2；"
                     "T_det 只計入有被偵測到的失敗 episode；長度基準 = 只用時間 t 當分數（不上色）；"
                     f"綠色較好、紅色較差；數值為 {args.repeats} 次重複 × {args.folds} 折的平均。")

    suites = list(results[0][1]["suite"])
    ws = wb.create_sheet(f"Suites α={args.alpha:g}")
    rows = [[layer] + [v for s in suites for v in (r["suite"][s]["BA"], r["suite"][s]["early_score"],
                                                   r["suite"][s]["T_det_mean"])] for layer, r in results]
    write_sheet(ws, ["Layer"] + [f"{s}\n{h}" for s in suites for h in ("Balanced Acc.", "早期偵測分數", "平均 T_det")],
                rows, ["0"] + ["0.000", "0.000", "0.0"] * len(suites), [8] + [12, 12, 11] * len(suites))
    for j in range(len(suites)):
        for d, hb in [(0, True), (1, True), (2, False)]:
            color(ws, get_column_letter(2 + 3 * j + d), len(rows), hb)
    add_note(ws, f"α={args.alpha:g}；各 suite 共用整體校準的門檻。")

    ts = time_points(data["n_chunks"])
    ws = wb.create_sheet("AUROC by time")
    write_sheet(ws, ["Layer"] + [f"t={t}" for t in ts], [[layer, *r["time"]] for layer, r in results],
                ["0"] + ["0.000"] * len(ts), [8] + [9] * len(ts))
    for j in range(len(ts)):
        color(ws, get_column_letter(2 + j), len(results))
    add_note(ws, "第 t 個 chunk 時，只用當下分數區分仍在進行的成功 / 失敗 episode 的 AUROC（與門檻無關）。")

    write_sheet(wb.create_sheet("Settings"), ["參數", "值"], settings_rows(args, 特徵=feature), [None, None], [16, 28])
    wb.save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feature", default="lm_mean", choices=["lm_last", "lm_mean", "act_mean", "act_prelogits"],
                   help="lm_mean / lm_last = prefill（影像 + 指令）；act_mean = 解碼階段產生動作 token 位置的平均（18 層）；"
                        "act_prelogits = 同上但為 final RMSNorm 之後（只有 1 層，--layer 0）")
    p.add_argument("--layer", type=int, default=12)
    p.add_argument("--sweep", action="store_true", help="逐層評估，只印 --alpha 下的摘要")
    p.add_argument("--alphas", type=str, default="0.05,0.1,0.2", help="可接受的成功誤報率")
    p.add_argument("--alpha", type=float, default=0.1, help="各 suite 細項與 sweep 摘要使用的 alpha")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--input_dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--score", default="cumsum", choices=["prefix", "cumsum"],
                   help="prefix = 前綴平均特徵 -> MLP logit；cumsum = 逐 chunk sigmoid(MLP) 的累加分數（同 SAFE-MLP）")
    p.add_argument("--input", default="sub_first", choices=["raw", "sub_first", "delta", "first"],
                   help="去除初始圖像 / 語言等靜態特徵：raw = 原始；sub_first = e_t - e_0；delta = e_t - e_{t-1}；"
                        "first = 診斷用，只給 e_0")
    p.add_argument("--window", type=int, default=0,
                   help="prefix：只平均最近 window 個 chunk（0 = 全部前綴），避免一開始的 chunk 一直留在輸入中")
    p.add_argument("--skip", type=int, default=0,
                   help="略過每個 episode 的前 skip 個 chunk：不用它們的特徵，也不在這段時間警報"
                        "（sub_first / delta 至少為 1，因為第 0 個 chunk 的輸入恆為 0）")
    p.add_argument("--cumsum_bce", type=float, default=0.0,
                   help="cumsum：每個 episode 另加 λ·Σ_t 逐 chunk BCE，避免 sigmoid 飽和（0 = 同 SAFE）")
    p.add_argument("--epochs", type=int, default=None, help="預設 prefix 為 30，cumsum 為 1000（同 SAFE）")
    p.add_argument("--batch_size", type=int, default=None,
                   help="預設 prefix 為 256 個前綴樣本，cumsum 為 512 個 episode（同 SAFE）")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--calib_folds", type=int, default=4, help="訓練折內校準門檻用的 inner CV 折數")
    p.add_argument("--split", default="episode", choices=["episode", "task"],
                   help="episode = 隨機切分 episode（看過的任務）；task = 以基礎任務為單位切分（未見任務）")
    p.add_argument("--threshold", default=None, choices=["const", "functional"],
                   help="預設 cumsum 為 functional、prefix 為 const；const = 固定門檻；functional = 隨時間變化的門檻 μ(t)+h·m(t)（functional CP）")
    p.add_argument("--fcp_modulation", default="tfunc", choices=["std", "tfunc"],
                   help="functional：調變函數 m(t)；std = 逐時間標準差，tfunc = 去掉極端曲線後的逐時間最大偏差（同 SAFE）")
    p.add_argument("--fcp_align", default="extend", choices=["freeze", "extend"],
                   help="functional：提早結束的成功 episode；freeze = 樣本不足處沿用最後可靠的 μ、m，"
                        "extend = 以最後一個分數補到 Tmax（同 SAFE）")
    p.add_argument("--fcp_fit_frac", type=float, default=0.3,
                   help="functional：用來估計 μ、m 的成功 episode 比例，其餘用來校準 h（SAFE 為 0.3）")
    p.add_argument("--fcp_min_count", type=int, default=10,
                   help="functional（freeze）：估計 μ(t)、m(t) 時，每個時間點至少要有的成功 episode 數")
    p.add_argument("--deadline", default="min", choices=["none", "min", "median"],
                   help="偵測期限：none = 整個 episode；min / median = 同一基礎任務成功 episode 長度的最小值 / 中位數"
                        "（同 SAFE 的 by earliest stop），期限後的警報不計，門檻也只用期限內的分數校準")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--xlsx", type=str, default=None, help="把結果另存成試算表（.xlsx）；--sweep 時為逐層比較表")
    args = p.parse_args()
    if args.threshold is None:
        args.threshold = "functional" if args.score == "cumsum" else "const"
    if args.epochs is None:
        args.epochs = 30 if args.score == "prefix" else 1000
    if args.batch_size is None:
        args.batch_size = 256 if args.score == "prefix" else 512
    args.alphas = [float(a) for a in args.alphas.split(",")]
    if args.alpha not in args.alphas:
        args.alphas.append(args.alpha)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_dataset()
    data["deadline"] = detection_deadline(data, args.deadline)
    y, n = data["label"], data["n_chunks"]
    print(f"episodes={len(y)}  成功={y.sum()}  失敗={(~y).sum()}  最短 episode={n.min()} chunks  "
          f"偵測期限={args.deadline}（失敗 episode 平均期限 {data['deadline'][~y].mean():.1f} chunks）")
    F = load_chunk_features(args.feature, data)
    bad = nonfinite_layers(F)
    if not args.sweep and args.layer in bad:
        sys.exit(f"第 {args.layer} 層特徵含 inf / NaN，請改用其他層")
    base = length_baseline(data, args)

    if args.sweep:
        a = args.alphas.index(args.alpha)
        base_r = layer_results(*base, data, args)
        m = base_r["alpha"][a]
        print(f"長度基準: 早期偵測分數={m['early_score']:.3f}  偵測率={m['TPR']:.3f}  "
              f"誤報率={m['FPR']:.3f}  T_det 中位數={m['T_det_median']:.1f}  佔 timeout={m['frac_median']:.2f}", flush=True)
        results = []
        for layer in range(F.shape[0]):
            if layer in bad:
                continue
            scores, thr = run_layer(F[layer], data, args, device)
            r =layer_results(scores, thr, data, args)
            results.append((layer, r))
            if args.xlsx:  # 每層跑完就更新，中斷時保留已完成的層
                save_sweep_xlsx(args.xlsx, args.feature, results, data, args, base_r)
            m = r["alpha"][a]
            print(f"{args.feature} layer {layer:2d}: 早期偵測分數={m['early_score']:.3f}  偵測率={m['TPR']:.3f}  "
                  f"誤報率={m['FPR']:.3f}  T_det 中位數={m['T_det_median']:.1f}  佔 timeout={m['frac_median']:.2f}", flush=True)
        print(f"前 5 名（早期偵測分數, alpha={args.alpha}）:",
              "  ".join(f"L{l}={r['alpha'][a]['early_score']:.3f}"
                        for l, r in sorted(results, key=lambda x: -x[1]["alpha"][a]["early_score"])[:5]))
        if args.xlsx:
            print(f"已寫出 {args.xlsx}")
        return
    scores, thr = run_layer(F[args.layer], data, args, device)
    report(f"{args.feature} layer {args.layer}", scores, thr, data, args, base)
    if args.xlsx:
        save_xlsx(args.xlsx, f"{args.feature} layer {args.layer}", scores, thr, data, args, base)


if __name__ == "__main__":
    main()
