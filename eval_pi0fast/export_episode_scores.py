"""跑一次 online_failure_detector（單層模式），把逐 episode 的偵測結果存成 CSV + NPZ，供人工對照影片檢查。

除了下列選項，其餘參數都原樣交給 online_failure_detector（例如 --layer、--threshold、--repeats），
流程與數值完全相同（透過攔截 report 取得 scores / thresholds，不修改原程式）。

輸出：
  <out>.csv  每個 episode 一列：T_det（已看過的 chunk 數）、alarm_chunk（= T_det - 1，即影片上標示的 chunk 編號）、
             各 repeat 中被警報的次數、期限內最大分數與該時間點的門檻、逐 chunk 分數 / 門檻（以空白分隔）
  <out>.npz  完整陣列：scores (R, N, Tmax)、thr (R, N, Tmax)（--alpha 下）、deadline、n_chunks、以及對應的 key

用法：
    python export_episode_scores.py --out outputs/episode_scores_L10 --layer 10 --threshold functional
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

import online_failure_detector as ofd


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--out", type=str, default=str(ofd.ROOT / "episode_scores"), help="輸出路徑（不含副檔名）")
    p.add_argument("--repeat", type=int, default=0, help="CSV 中 T_det / 分數取自哪一個 repeat")
    own, rest = p.parse_known_args()
    if "--sweep" in rest:
        sys.exit("只支援單層模式，請勿加 --sweep")

    captured = {}
    orig_report = ofd.report

    def report(name, scores, thr, data, args, base):
        captured.update(scores=scores, thr=thr, data=data, args=args)
        orig_report(name, scores, thr, data, args, base)

    ofd.report = report
    sys.argv = [sys.argv[0], *rest]
    ofd.main()

    scores, data, args = captured["scores"], captured["data"], captured["args"]
    thr = captured["thr"][:, args.alphas.index(args.alpha)]  # (R, N, Tmax)
    y, n, d = data["label"], data["n_chunks"], data["deadline"]
    s_dl = ofd.within_deadline(scores.reshape(-1, scores.shape[-1]), np.tile(d, len(scores))).reshape(scores.shape)
    alarm = ofd.is_alarm(s_dl, thr)  # (R, N, Tmax)，同 detection_metrics
    detected = alarm.any(2)
    t_det = alarm.argmax(2) + 1
    r = own.repeat

    rows = list(csv.DictReader(open(ofd.ROOT / "results.csv")))  # 順序同 load_dataset
    out = Path(own.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["suite", "task_id", "episode", "task_name", "category", "success", "n_chunks", "deadline",
                    "detected", "T_det", "alarm_chunk", "detected_repeats", "max_score", "thr_at_max",
                    "scores", "thresholds"])
        for i, row in enumerate(rows):
            s, tau = s_dl[r, i, : n[i]], thr[r, i, : n[i]]
            k = int(np.nanargmax(s))
            w.writerow([row["suite"], row["task_id"], row["episode"], row["task_name"], row["category"], bool(y[i]),
                        int(n[i]), int(d[i]), bool(detected[r, i]),
                        int(t_det[r, i]) if detected[r, i] else "", int(t_det[r, i]) - 1 if detected[r, i] else "",
                        int(detected[:, i].sum()), f"{s[k]:.4f}", f"{tau[k]:.4f}",
                        " ".join("nan" if np.isnan(v) else f"{v:.4f}" for v in scores[r, i, : n[i]]),
                        " ".join(f"{v:.4f}" for v in tau)])
    np.savez_compressed(out.with_suffix(".npz"), scores=scores, thr=thr, deadline=d, n_chunks=n, label=y,
                        key=np.array([f"{r['suite']}_{r['task_id']}_ep{r['episode']}" for r in rows]),
                        alpha=args.alpha, repeat=r)

    f = ~y
    print(f"\nrepeat {r}（α={args.alpha}）：失敗偵測率={detected[r, f].mean():.3f}  成功誤報率={detected[r, y].mean():.3f}  "
          f"T_det 中位數={np.median(t_det[r, f & detected[r]]):.1f}")
    print(f"已寫出 {out.with_suffix('.csv')} 與 {out.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
