"""把 export_episode_scores.py 的偵測結果疊到評估影片上，方便人工判斷警報是否過早。

每支影片：上方為原始畫面（左上角標 chunk / step 與狀態），下方為分數曲線面板：
  白線 = 失敗分數 s_t（只畫到目前的 chunk，即線上當下看得到的部分）、黃虛線 = 門檻 τ(t)、
  灰色直線 = 偵測期限、紅色直線 = 警報時間。第 c 個 chunk 的分數在 step[c]（該 chunk 開始時）就可取得，
  所以警報在 step[alarm_chunk] 那一幀出現，之後畫面加紅框。

自動挑選的 episode（依 --repeat 那次的結果）：
  fail_early   有偵測到的失敗 episode 中 T_det 最小的 --n_early 個（檢查是否過早）
  false_alarm  被誤報的成功 episode（最多 --n_fa 個，依 T_det 由早到晚）
  fail_late    T_det 位於中位數附近的失敗 episode（--n_other 個，對照用）
  fail_missed  未被偵測到的失敗 episode（--n_other 個）
  success_ok   未被誤報的成功 episode（--n_other 個）

用法：
    python export_episode_scores.py --out outputs/episode_scores_cumsum_L12
    python render_detection_videos.py --scores outputs/episode_scores_cumsum_L12.csv
    python render_detection_videos.py --scores outputs/episode_scores_prefix_L12.csv \
        --same_as outputs/annotated_videos/episode_scores_cumsum_L12/index.csv
（tag 為原資料夾的分類；同名 episode 在不同資料夾中可直接比對）
"""

import argparse
import csv
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from annotate_chunks import ROOT, find_one

PANEL_H = 160


def parse(row):
    vals = lambda s: np.array([float(v) for v in s.split()])
    return dict(row, n_chunks=int(row["n_chunks"]), deadline=int(row["deadline"]),
                success=row["success"] == "True", detected=row["detected"] == "True",
                alarm_chunk=int(row["alarm_chunk"]) if row["alarm_chunk"] else None,
                s=vals(row["scores"]), tau=vals(row["thresholds"]))


def select(rows, args):
    fail = [r for r in rows if not r["success"]]
    det = sorted([r for r in fail if r["detected"]], key=lambda r: r["alarm_chunk"])
    pick = [("fail_early", r) for r in det[: args.n_early]]
    pick += [("false_alarm", r) for r in sorted([r for r in rows if r["success"] and r["detected"]],
                                                 key=lambda r: r["alarm_chunk"])[: args.n_fa]]
    mid = len(det) // 2
    pick += [("fail_late", r) for r in det[max(mid, args.n_early): max(mid, args.n_early) + args.n_other]]
    rng = np.random.default_rng(0)
    for tag, pool in [("fail_missed", [r for r in fail if not r["detected"]]),
                      ("success_ok", [r for r in rows if r["success"] and not r["detected"]])]:
        pick += [(tag, pool[i]) for i in rng.permutation(len(pool))[: args.n_other]]
    return pick


def panel(r, cur, width):
    """分數曲線面板 (PANEL_H, width, 3)，cur = 目前的 chunk 編號。"""
    img = np.full((PANEL_H, width, 3), 30, np.uint8)
    s, tau, n = r["s"], r["tau"], r["n_chunks"]
    fin = np.isfinite(tau)
    lo = min(np.nanmin(s), tau[fin].min() if fin.any() else np.nanmin(s))
    hi = max(np.nanmax(s), tau[fin].max() if fin.any() else np.nanmax(s))
    lo, hi = lo - 0.05 * (hi - lo + 1e-6), hi + 0.1 * (hi - lo + 1e-6)
    x0, x1, y0, y1 = 44, width - 8, 22, PANEL_H - 18
    X = lambda c: int(x0 + (x1 - x0) * c / max(n - 1, 1))
    Y = lambda v: int(y1 - (y1 - y0) * (v - lo) / (hi - lo))
    cv2.rectangle(img, (x0, y0), (x1, y1), (80, 80, 80), 1)
    txt = lambda t, org, col, sc=0.42: cv2.putText(img, t, org, cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1, cv2.LINE_AA)
    for v in np.linspace(lo, hi, 4)[:3] + (hi - lo) / 6:  # 縱軸刻度（分數 / 門檻的數值）
        cv2.line(img, (x0 - 3, Y(v)), (x1, Y(v)), (55, 55, 55), 1)
        txt(f"{v:.1f}", (2, Y(v) + 4), (180, 180, 180), 0.36)
    step = 5 if n <= 30 else 10
    for c in range(0, n, step):  # 橫軸刻度（chunk 編號）
        cv2.line(img, (X(c), y1), (X(c), y1 + 3), (180, 180, 180), 1)
        txt(str(c), (X(c) - 4, PANEL_H - 4), (180, 180, 180), 0.36)
    if r["deadline"] < n:
        cv2.line(img, (X(r["deadline"]), y0), (X(r["deadline"]), y1), (140, 140, 140), 1)
    for c in range(n - 1):  # 門檻（虛線）
        if fin[c] and fin[c + 1] and c % 2 == 0:
            cv2.line(img, (X(c), Y(tau[c])), (X(c + 1), Y(tau[c + 1])), (255, 220, 0), 1, cv2.LINE_AA)
    for c in range(min(cur, n - 1)):  # 分數只畫到目前的 chunk
        if np.isfinite(s[c]) and np.isfinite(s[c + 1]):
            cv2.line(img, (X(c), Y(s[c])), (X(c + 1), Y(s[c + 1])), (255, 255, 255), 2, cv2.LINE_AA)
    if np.isfinite(s[cur]):
        cv2.circle(img, (X(cur), Y(s[cur])), 3, (255, 255, 255), -1)
    a = r["alarm_chunk"]
    if a is not None and cur >= a:
        cv2.line(img, (X(a), y0), (X(a), y1), (255, 60, 60), 2)
    txt(f"s={s[cur]:.2f}  tau={tau[cur]:.2f}" if np.isfinite(tau[cur]) and np.isfinite(s[cur])
        else "(skipped chunk)", (x0, 15), (255, 255, 255))
    txt("score", (x1 - 150, 15), (255, 255, 255))
    txt("thr", (x1 - 100, 15), (255, 220, 0))
    txt("deadline", (x1 - 65, 15), (140, 140, 140))
    return img


def render(tag, r, args, out_dir):
    name = f"{r['suite']}_{r['task_id']}"
    video = find_one(f"{r['suite']}/chunk_*/videos/{name}/eval_episode_{r['episode']}.mp4")
    step = np.load(find_one(f"{r['suite']}/chunk_*_features/{name}_ep{r['episode']}.npz"))["step"]
    frames = imageio.mimread(video, memtest=False)
    chunk = np.searchsorted(step, np.arange(len(frames)), side="right") - 1
    a = r["alarm_chunk"]
    head = ("SUCCESS" if r["success"] else "FAILURE") + f" | {tag}"
    out = []
    for i in range(0, len(frames), args.stride):
        f, c = frames[i].copy(), int(min(chunk[i], r["n_chunks"] - 1))
        alarmed = a is not None and c >= a
        if alarmed:
            cv2.rectangle(f, (0, 0), (f.shape[1] - 1, f.shape[0] - 1), (255, 0, 0), 6)
        lines = [(head, (255, 255, 255)), (f"chunk {c}/{r['n_chunks'] - 1}  step {i}", (255, 255, 0)),
                 (f"ALARM @ chunk {a}" if alarmed else "no alarm", (255, 80, 80) if alarmed else (120, 255, 120))]
        for j, (t, col) in enumerate(lines):
            org = (8, 20 + 20 * j)
            cv2.putText(f, t, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(f, t, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        out.append(np.vstack([f, panel(r, c, f.shape[1])]))
    t_str = f"a{a:02d}" if a is not None else "noalarm"
    path = out_dir / f"{tag}_{t_str}_{name}_ep{r['episode']}.mp4"
    imageio.mimwrite(path, out, fps=args.fps / args.stride * 1.0, macro_block_size=8)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True, help="export_episode_scores.py 輸出的 CSV")
    p.add_argument("--out_dir", default=None, help="預設 outputs/annotated_videos/<CSV 檔名>")
    p.add_argument("--n_early", type=int, default=10)
    p.add_argument("--n_fa", type=int, default=6)
    p.add_argument("--n_other", type=int, default=3)
    p.add_argument("--fps", type=float, default=20, help="輸出播放速度（原始影片為 80 fps）")
    p.add_argument("--same_as", type=str, nargs="*", default=[],
                   help="另一個資料夾的 index.csv：把同一批 episode 也畫一份到 <out_dir>/same_as_<該資料夾名>，方便並排比對")
    p.add_argument("--stride", type=int, default=1, help="每 stride 幀取一幀（縮小檔案）")
    args = p.parse_args()

    rows = [parse(r) for r in csv.DictReader(open(args.scores))]
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "annotated_videos" / Path(args.scores).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(out_dir, select(rows, args))]
    by_key = {(r["suite"], r["task_id"], r["episode"]): r for r in rows}
    for idx in args.same_as:
        other = list(csv.DictReader(open(idx)))
        jobs.append((out_dir / f"same_as_{Path(idx).parent.name}",
                     [(o["tag"], by_key[(o["suite"], o["task_id"], o["episode"])]) for o in other]))
    for d, picks in jobs:
        d.mkdir(parents=True, exist_ok=True)
        summary = []
        for tag, r in picks:
            path = render(tag, r, args, d)
            summary.append([path.name, tag, r["suite"], r["task_id"], r["episode"], r["task_name"], r["success"],
                            r["n_chunks"], r["deadline"], r["alarm_chunk"] if r["alarm_chunk"] is not None else ""])
            print(f"已寫出 {path}", flush=True)
        with open(d / "index.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["video", "tag", "suite", "task_id", "episode", "task_name", "success", "n_chunks", "deadline",
                        "alarm_chunk"])
            w.writerows(summary)
        print(f"清單：{d / 'index.csv'}")


if __name__ == "__main__":
    main()
