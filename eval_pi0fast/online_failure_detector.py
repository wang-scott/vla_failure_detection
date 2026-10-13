"""線上失敗偵測：每個 chunk 依「到目前為止」的特徵給失敗分數，評估多早能發現失敗。

流程（每個 outer fold）：
  1. 分數：時間 t 的輸入是前 t+1 個 chunk 特徵的平均（嚴格因果），MLP 輸出失敗 logit s_t。
     訓練時每個 episode 的所有前綴都是樣本，每個 episode 總權重相同，再做類別平衡。
  2. 門檻：在訓練折內做 inner CV，取得訓練折中成功 episode 的 out-of-fold 分數，
     以「每個成功 episode 的最大分數」做 conformal 分位數，使成功 episode
     在任何時刻被誤報的比例 <= alpha。
  3. 偵測：測試折中第一次 s_t > 門檻的 chunk 即為偵測時間 T_det。

失敗 episode 都跑到 timeout，沒有失敗起點的標註，所以偵測時間以「已看過的 chunk 數」
及「佔 timeout 的比例」表示；早期偵測分數 = 失敗 episode 平均剩餘時間比例（未偵測算 0）。

用法：
    python online_failure_detector.py --feature lm_mean --layer 10
    python online_failure_detector.py --feature lm_mean --sweep --repeats 1
"""

import argparse
import csv
import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).parent / "outputs"


def load_dataset():
    """依 outputs/results.csv 的順序，回傳 label (N,)（True = 成功）、n_chunks (N,)、suite (N,)、category (N,)。"""
    rows = list(csv.DictReader(open(ROOT / "results.csv")))
    feat_files = {Path(f).stem: f for f in glob.glob(str(ROOT / "*/chunk_*_features/*.npz"))}
    label_files = {Path(f).stem: f for f in glob.glob(str(ROOT / "*/chunk_*_bddl/*.npz"))}
    out = {"label": [], "n_chunks": [], "suite": [], "category": []}
    for r in rows:
        key = f"{r['suite']}_{r['task_id']}_ep{r['episode']}"
        out["label"].append(bool(np.load(label_files[key])["label"][0]))
        out["n_chunks"].append(len(np.load(feat_files[key])["step"]))
        out["suite"].append(r["suite"])
        out["category"].append(r["category"])
    return {k: np.array(v) for k, v in out.items()}


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
    """回傳 (L, N, Tmax, D) float16 memmap，時間超過 n_chunks 的部分為 0。順序同 load_dataset。"""
    cache = ROOT / f"chunk_features_{feature}.npy"
    if cache.exists():
        return np.load(cache, mmap_mode="r")
    rows = list(csv.DictReader(open(ROOT / "results.csv")))
    feat_files = {Path(f).stem: f for f in glob.glob(str(ROOT / "*/chunk_*_features/*.npz"))}
    first = np.load(feat_files[f"{rows[0]['suite']}_{rows[0]['task_id']}_ep{rows[0]['episode']}"])[feature]
    n_layers = 54 if feature == "vision_mean" else first.shape[1]
    tmp = cache.with_suffix(".tmp.npy")  # 寫完才改名，中斷時不會留下壞掉的快取
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16,
                                    shape=(n_layers, len(rows), int(data["n_chunks"].max()), first.shape[-1]))
    for i, r in enumerate(rows):
        f = np.load(feat_files[f"{r['suite']}_{r['task_id']}_ep{r['episode']}"])[feature]
        if feature == "vision_mean":  # 只取前兩個相機，第 3 個是空白相機
            f = f[:, :2].reshape(len(f), -1, f.shape[-1])
        out[:, i, : len(f)] = f.transpose(1, 0, 2)
    out.flush()
    del out
    tmp.rename(cache)
    return np.load(cache, mmap_mode="r")


def prefix_means(f, n):
    """f (N, Tmax, D) -> 每個時間點前綴平均 (N, Tmax, D)。"""
    cs = np.cumsum(f.astype(np.float32), axis=1)
    return cs / np.arange(1, f.shape[1] + 1, dtype=np.float32)[None, :, None]


def fit_predict(P, n, fail, tr, te, args, device):
    """用 tr 中 episode 的所有前綴訓練，回傳 te 的逐 chunk 失敗分數 (len(te), Tmax)，超出長度為 nan。"""
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


def conformal_threshold(success_max, alpha):
    """使新的成功 episode 最大分數超過門檻的機率 <= alpha。"""
    m = np.sort(success_max)
    k = int(np.ceil((len(m) + 1) * (1 - alpha)))
    return np.inf if k > len(m) else m[k - 1]


def run_online(P, data, args, device):
    """回傳 scores (repeats, N, Tmax) 與 thresholds (repeats, A, N)：每個 episode 在其測試折的門檻。"""
    y, n = data["label"], data["n_chunks"]
    fail = ~y
    strata = np.char.add(data["suite"].astype(str), y.astype(str))
    scores = np.full((args.repeats, len(y), P.shape[1]), np.nan)
    thr = np.zeros((args.repeats, len(args.alphas), len(y)))
    for seed in range(args.repeats):
        torch.manual_seed(seed)
        fold = stratified_folds(strata, args.folds, seed)
        for k in range(args.folds):
            tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
            inner = stratified_folds(strata[tr], args.calib_folds, seed + 1000)
            success_max = []
            for j in range(args.calib_folds):
                itr, ite = tr[inner != j], tr[inner == j]
                ite = ite[y[ite]]  # 只需成功 episode 的分數
                success_max.append(np.nanmax(fit_predict(P, n, fail, itr, ite, args, device), axis=1))
            success_max = np.concatenate(success_max)
            for a, alpha in enumerate(args.alphas):
                thr[seed, a, te] = conformal_threshold(success_max, alpha)
            scores[seed, te] = fit_predict(P, n, fail, tr, te, args, device)
    return scores, thr


def detection_metrics(s, tau, data, mask=None):
    """s (N, Tmax) 失敗分數、tau (N,) 門檻 -> 指標 dict。"""
    y, n = data["label"], data["n_chunks"]
    if mask is not None:
        s, tau, y, n = s[mask], tau[mask], y[mask], n[mask]
    alarm = np.nan_to_num(s, nan=-np.inf) > tau[:, None]
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
    return float(np.mean([auroc(~y[n >= t], s[n >= t, t - 1]) for s in scores]))


def report(name, scores, thr, data, args):
    y, n = data["label"], data["n_chunks"]
    print(f"[{name}]")
    max_auroc = np.mean([auroc(~y, np.nanmax(s, 1)) for s in scores])
    print(f"  episode 層級 AUROC（最大失敗分數）= {max_auroc:.3f}")
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
    rows = [(f"all α={alpha:.2f}", a, None) for a, alpha in enumerate(args.alphas)]
    rows += [(f"{s} α={args.alpha:.2f}", a, data["suite"] == s) for s in np.unique(data["suite"])]
    for label, ai, mask in rows:
        m = mean_metrics(scores, thr, ai, data, mask)
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


def save_xlsx(path, name, scores, thr, data, args):
    """把摘要、累積偵測率、逐時間 AUROC 與設定寫成試算表；BA 以公式計算。"""
    from openpyxl import Workbook

    y, n = data["label"], data["n_chunks"]
    scopes = [("all", None)] + [(s, data["suite"] == s) for s in np.unique(data["suite"])]
    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    rows, cum_rows = [], []
    for a, alpha in enumerate(args.alphas):
        for scope, mask in scopes:
            mk = np.ones(len(y), bool) if mask is None else mask
            m = mean_metrics(scores, thr, a, data, mask)
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
                 f"數值為 {args.repeats} 次重複 × {args.folds} 折的平均。")

    write_sheet(wb.create_sheet("Cumulative"), ["Scope", "α"] + [f"≤{q:.0%} timeout" for q in np.linspace(0.1, 1, 10)],
                cum_rows, [None, "0.00"] + ["0.000"] * 10, [16, 6] + [11] * 10)

    write_sheet(wb.create_sheet("AUROC by time"), ["t (chunks)", "仍在進行的 episodes", "AUROC"],
                [[t, int((n >= t).sum()), auroc_at(scores, y, n, t)] for t in time_points(n)]
                + [["episode 層級（最大分數）", len(y), float(np.mean([auroc(~y, np.nanmax(s, 1)) for s in scores]))]],
                [None, "0", "0.000"], [24, 18, 10])

    write_sheet(wb.create_sheet("Settings"), ["參數", "值"], settings_rows(args, 模型=name), [None, None], [16, 28])
    wb.save(path)
    print(f"\n已寫出 {path}")


def layer_results(scores, thr, data, args):
    """單一層的所有指標：各 alpha 的整體指標、--alpha 下各 suite 指標、逐時間 AUROC。"""
    y, n = data["label"], data["n_chunks"]
    a = args.alphas.index(args.alpha)
    return {
        "auroc": float(np.mean([auroc(~y, np.nanmax(s, 1)) for s in scores])),
        "alpha": [mean_metrics(scores, thr, i, data) for i in range(len(args.alphas))],
        "suite": {s: mean_metrics(scores, thr, a, data, data["suite"] == s) for s in np.unique(data["suite"])},
        "time": [auroc_at(scores, y, n, t) for t in time_points(n)],
    }


def save_sweep_xlsx(path, feature, results, data, args):
    """逐層比較表：每個 alpha 一張（每列一層），另有各 suite、逐時間 AUROC、設定。"""
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
        for layer, r in results:
            m, k = r["alpha"][i], len(rows) + 2
            rows.append([layer, r["auroc"], m["TPR"], m["FPR"], f"=(C{k}+1-D{k})/2", m["early_score"],
                         m["T_det_mean"], m["T_det_median"], m["frac_mean"], m["frac_median"]])
        write_sheet(ws, ["Layer", "Episode AUROC", "TPR（失敗偵測率）", "FPR（成功誤報率）", "Balanced Acc.",
                         "早期偵測分數", "平均 T_det (chunks)", "T_det 中位數 (chunks)", "平均佔 timeout", "佔 timeout 中位數"],
                    rows, ["0", "0.000", "0.000", "0.000", "0.000", "0.000", "0.0", "0.0", "0.00", "0.00"],
                    [8, 11, 12, 12, 11, 11, 12, 12, 11, 11])
        for col, hb in [("B", True), ("C", True), ("D", False), ("E", True), ("F", True), ("G", False), ("I", False)]:
            color(ws, col, len(rows), hb)
        add_note(ws, f"{feature}，α={alpha:g}。Balanced Acc. = (TPR + 1 − FPR) / 2；T_det 只計入有被偵測到的失敗 episode；"
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
    p.add_argument("--feature", default="lm_mean", choices=["lm_last", "lm_mean", "vision_mean"])
    p.add_argument("--layer", type=int, default=10, help="vision_mean 的層編號為 cam*27+layer（0~53）")
    p.add_argument("--sweep", action="store_true", help="逐層評估，只印 --alpha 下的摘要")
    p.add_argument("--alphas", type=str, default="0.05,0.1,0.2", help="可接受的成功誤報率")
    p.add_argument("--alpha", type=float, default=0.1, help="各 suite 細項與 sweep 摘要使用的 alpha")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--input_dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--calib_folds", type=int, default=4, help="訓練折內校準門檻用的 inner CV 折數")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--xlsx", type=str, default=None, help="把結果另存成試算表（.xlsx）；--sweep 時為逐層比較表")
    args = p.parse_args()
    args.alphas = [float(a) for a in args.alphas.split(",")]
    if args.alpha not in args.alphas:
        args.alphas.append(args.alpha)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_dataset()
    y, n = data["label"], data["n_chunks"]
    print(f"episodes={len(y)}  成功={y.sum()}  失敗={(~y).sum()}  最短 episode={n.min()} chunks")
    F = load_chunk_features(args.feature, data)

    if args.sweep:
        a = args.alphas.index(args.alpha)
        results = []
        for layer in range(F.shape[0]):
            scores, thr = run_online(prefix_means(F[layer], n), data, args, device)
            r = layer_results(scores, thr, data, args)
            results.append((layer, r))
            if args.xlsx:  # 每層跑完就更新，中斷時保留已完成的層
                save_sweep_xlsx(args.xlsx, args.feature, results, data, args)
            m = r["alpha"][a]
            print(f"{args.feature} layer {layer:2d}: 早期偵測分數={m['early_score']:.3f}  偵測率={m['TPR']:.3f}  "
                  f"誤報率={m['FPR']:.3f}  T_det 中位數={m['T_det_median']:.1f}  佔 timeout={m['frac_median']:.2f}", flush=True)
        print(f"前 5 名（早期偵測分數, alpha={args.alpha}）:",
              "  ".join(f"L{l}={r['alpha'][a]['early_score']:.3f}"
                        for l, r in sorted(results, key=lambda x: -x[1]["alpha"][a]["early_score"])[:5]))
        if args.xlsx:
            print(f"已寫出 {args.xlsx}")
        return
    scores, thr = run_online(prefix_means(F[args.layer], n), data, args, device)
    report(f"{args.feature} layer {args.layer}", scores, thr, data, args)
    if args.xlsx:
        save_xlsx(args.xlsx, f"{args.feature} layer {args.layer}", scores, thr, data, args)


if __name__ == "__main__":
    main()
