"""在 LIBERO-plus 上評測 pi0-FAST，並依擾動類別 / 難度統計成功率。

LIBERO-plus 每個 suite 約有 2400~2500 個擾動任務（7 種擾動類別 x 5 種難度），
全部跑完在單張 GPU 上需要數天，因此預設對每個擾動類別分層抽樣。

用法（先 `source <repo>/activate.sh`）：
    # 快速冒煙測試：1 個 suite、每類別 1 個任務
    python eval_pi0fast_libero_plus.py --suites libero_spatial --per-category 1

    # 抽樣評測：4 個 suite、每類別 20 個任務
    python eval_pi0fast_libero_plus.py --per-category 20

    # 完整評測（非常久）
    python eval_pi0fast_libero_plus.py --all

    # 只重新統計已有結果
    python eval_pi0fast_libero_plus.py --analyze-only

任務分批（--chunk-size）執行，每批結果各自寫入 eval_info.json；
中斷後以相同參數重跑會自動跳過已完成的批次。
"""

import argparse
import csv
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

VLA_ROOT = Path(__file__).resolve().parent.parent  # repo 根目錄
LIBERO_PLUS_ROOT = VLA_ROOT / "LIBERO-plus"
CLASSIFICATION_FILE = LIBERO_PLUS_ROOT / "libero/libero/benchmark/task_classification.json"
DEFAULT_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def load_classification() -> dict[str, list[dict]]:
    """suite -> 依 suite 內任務順序排列的 [{id, name, category, difficulty_level}]。

    分類檔的 id 從 1 開始，對應 lerobot 的 task_id = id - 1。
    """
    with open(CLASSIFICATION_FILE) as f:
        data = json.load(f)
    return {suite: sorted(tasks, key=lambda t: t["id"]) for suite, tasks in data.items()}


def select_tasks(classification, suites, per_category, use_all, seed) -> dict[str, list[int]]:
    rng = random.Random(seed)
    plan = {}
    for suite in suites:
        tasks = classification[suite]
        if use_all:
            plan[suite] = [t["id"] - 1 for t in tasks]
            continue
        by_category = defaultdict(list)
        for t in tasks:
            by_category[t["category"]].append(t["id"] - 1)
        chosen = []
        for category in sorted(by_category):
            ids = by_category[category]
            chosen += rng.sample(ids, min(per_category, len(ids)))
        plan[suite] = sorted(chosen)
    return plan


def run_chunk(args, suite, task_ids, chunk_dir: Path) -> bool:
    cmd = [
        *([sys.executable, str(Path(__file__).parent / "lerobot_eval_bddl.py")] if args.bddl_labels or args.save_features else ["lerobot-eval"]),
        f"--policy.path={args.policy}",
        "--env.type=libero_plus",
        f"--env.task={suite}",
        f"--env.task_ids=[{','.join(map(str, task_ids))}]",
        f"--eval.n_episodes={args.n_episodes}",
        "--eval.batch_size=1",
        "--env.max_parallel_tasks=1",
        f"--seed={args.seed}",
        f"--output_dir={chunk_dir}",
        "--policy.device=cuda",
        "--policy.gradient_checkpointing=false",
    ]
    if args.rename_map:
        cmd.append(f"--rename_map={args.rename_map}")
    cmd += args.extra

    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    if args.bddl_labels:
        env["BDDL_LABEL_DIR"] = str(label_dir(chunk_dir))
    if args.save_features:
        env["FEATURE_DIR"] = str(feature_dir(chunk_dir))
    # LIBERO-plus 以 namespace package 安裝，需放進 PYTHONPATH 才能正確 import
    env["PYTHONPATH"] = f"{LIBERO_PLUS_ROOT}:{env.get('PYTHONPATH', '')}"
    # torchvision 內建的 libjpeg.so.8 缺少 jpeg12_* 符號，會讓 Wand(ImageMagick) 載入失敗
    libjpeg = Path(sys.prefix) / "lib/libjpeg.so.8"
    if str(libjpeg) not in env.get("LD_PRELOAD", ""):
        env["LD_PRELOAD"] = f"{libjpeg}:{env.get('LD_PRELOAD', '')}".rstrip(":")
    # 伺服器沒有安裝 NVIDIA EGL 時，改用解壓到 <repo>/nvidia-gl 的 libnvidia-gl（存在才使用）
    nvidia_gl = VLA_ROOT / "nvidia-gl/root"
    if nvidia_gl.exists():
        env["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(nvidia_gl / "usr/share/glvnd/egl_vendor.d/10_nvidia.json")
        gl_lib = str(nvidia_gl / "usr/lib/x86_64-linux-gnu")
        if gl_lib not in env.get("LD_LIBRARY_PATH", ""):
            env["LD_LIBRARY_PATH"] = f"{gl_lib}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")

    log_path = chunk_dir.with_suffix(".log")
    print(f"  $ {' '.join(cmd)}\n  log: {log_path}", flush=True)
    # lerobot-eval 會拒絕已存在的 output_dir，因此只建立上層資料夾
    chunk_dir.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log:
        ret = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    if ret != 0:
        print(f"  !! 失敗 (exit {ret})，請查看 {log_path}", flush=True)
    return ret == 0


def run_eval(args, plan):
    out = Path(args.output_dir)
    failed = []
    for suite, task_ids in plan.items():
        chunks = [task_ids[i : i + args.chunk_size] for i in range(0, len(task_ids), args.chunk_size)]
        for i, chunk in enumerate(chunks):
            chunk_dir = out / suite / f"chunk_{i:04d}"
            if (chunk_dir / "eval_info.json").exists():
                print(f"[{suite}] chunk {i + 1}/{len(chunks)} 已完成，跳過")
                continue
            # 上次中斷留下的不完整結果
            for d in (chunk_dir, label_dir(chunk_dir), feature_dir(chunk_dir)):
                if d.exists():
                    subprocess.run(["rm", "-rf", str(d)], check=True)
            print(f"[{suite}] chunk {i + 1}/{len(chunks)}：{len(chunk)} 個任務", flush=True)
            if not run_chunk(args, suite, chunk, chunk_dir):
                failed.append(str(chunk_dir))
    return failed


def label_dir(chunk_dir: Path) -> Path:
    # lerobot-eval 不接受已存在的 output_dir，因此標籤放在旁邊的資料夾
    return chunk_dir.with_name(chunk_dir.name + "_bddl")


def feature_dir(chunk_dir: Path) -> Path:
    return chunk_dir.with_name(chunk_dir.name + "_features")


def load_labels(out: Path) -> dict:
    """(suite, task_id, episode) -> BDDL 成功標籤（bool）。"""
    import numpy as np

    labels = {}
    for path in out.glob("*/chunk_*_bddl/*.npz"):
        suite, task_id, ep = path.stem.rsplit("_", 2)  # libero_spatial_438_ep0
        labels[(suite, int(task_id), int(ep.removeprefix("ep")))] = bool(np.load(path)["label"][0])
    return labels


def pct(successes: list[bool]) -> str:
    return f"{100 * sum(successes) / len(successes):5.1f}% ({sum(successes)}/{len(successes)})"


def analyze(args, classification):
    out = Path(args.output_dir)
    labels = load_labels(out)
    rows = []
    for info_path in sorted(out.glob("*/chunk_*/eval_info.json")):
        with open(info_path) as f:
            info = json.load(f)
        for entry in info.get("per_task", []):
            suite, task_id = entry["task_group"], int(entry["task_id"])
            meta = classification[suite][task_id]
            for ep, success in enumerate(entry["metrics"]["successes"]):
                lab = labels.get((suite, task_id, ep), "")
                rows.append(
                    {
                        "suite": suite,
                        "task_id": task_id,
                        "task_name": meta["name"],
                        "category": meta["category"],
                        "difficulty": meta["difficulty_level"],
                        "episode": ep,
                        "success": bool(success),
                        "bddl_label": lab,
                    }
                )
    if not rows:
        print("找不到任何結果（eval_info.json）。")
        return

    with open(out / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def group(key):
        groups = defaultdict(list)
        for r in rows:
            groups[key(r)].append(r["success"])
        return dict(sorted(groups.items(), key=lambda kv: str(kv[0])))

    lines = [f"# pi0-FAST on LIBERO-plus\n", f"Policy: `{args.policy}`  ", f"Episodes: {len(rows)}\n"]
    lines.append(f"**Overall: {pct([r['success'] for r in rows])}**\n")

    lines.append("## By suite\n\n| Suite | Success |\n|---|---|")
    lines += [f"| {k} | {pct(v)} |" for k, v in group(lambda r: r["suite"]).items()]

    lines.append("\n## By perturbation category\n\n| Category | Success |\n|---|---|")
    lines += [f"| {k} | {pct(v)} |" for k, v in group(lambda r: r["category"]).items()]

    lines.append("\n## By difficulty level\n\n| Difficulty | Success |\n|---|---|")
    lines += [f"| {k} | {pct(v)} |" for k, v in group(lambda r: r["difficulty"]).items()]

    suites = sorted({r["suite"] for r in rows})
    categories = sorted({r["category"] for r in rows})
    cell = group(lambda r: (r["suite"], r["category"]))
    lines.append("\n## Suite x category\n")
    lines.append("| Category | " + " | ".join(suites) + " |")
    lines.append("|---" * (len(suites) + 1) + "|")
    for c in categories:
        vals = [pct(cell[(s, c)]) if (s, c) in cell else "-" for s in suites]
        lines.append(f"| {c} | " + " | ".join(vals) + " |")

    disagree = [r for r in rows if r["bddl_label"] != "" and r["bddl_label"] != r["success"]]
    if disagree:
        lines.append(f"\n⚠️ {len(disagree)} episodes: BDDL label 與 lerobot success 不一致")

    summary = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(summary)
    print("\n" + summary)
    print(f"逐筆結果：{out / 'results.csv'}\n摘要：{out / 'summary.md'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", default="lerobot/pi0fast-libero", help="HF repo id 或本地 checkpoint 路徑")
    p.add_argument("--suites", nargs="+", default=DEFAULT_SUITES, choices=DEFAULT_SUITES)
    p.add_argument("--per-category", type=int, default=10, help="每個 suite 每個擾動類別抽幾個任務")
    p.add_argument("--all", action="store_true", help="跑全部任務（忽略 --per-category）")
    p.add_argument("--n-episodes", type=int, default=1, help="每個任務跑幾次（LIBERO-plus 官方用 1）")
    p.add_argument("--chunk-size", type=int, default=20, help="每次呼叫 lerobot-eval 處理的任務數")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--rename-map", dest="rename_map", default="", help="觀測鍵名重新對應（JSON）")
    p.add_argument("--output-dir", default=str(VLA_ROOT / "eval_pi0fast/outputs"))
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--save-features", action="store_true", help="儲存 pi0-FAST 每層特徵（見 pi0fast_features.py）")
    p.add_argument("--bddl-labels", action="store_true", help="依 BDDL 目標條件為每個 rollout 產生成功/失敗標籤")
    p.add_argument("extra", nargs="*", help="其餘參數原樣傳給 lerobot-eval（放在 -- 之後）")
    args = p.parse_args()

    classification = load_classification()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not args.analyze_only:
        plan_path = out / "plan.json"
        plan = select_tasks(classification, args.suites, args.per_category, args.all, args.seed)
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            sys.exit(f"{out} 內已有不同參數的評測結果，請換一個 --output-dir。")
        plan_path.write_text(json.dumps(plan, indent=1))
        print(f"共 {sum(map(len, plan.values()))} 個任務 x {args.n_episodes} episode：" + ", ".join(f"{s}={len(t)}" for s, t in plan.items()))
        failed = run_eval(args, plan)
        if failed:
            print(f"\n{len(failed)} 個批次失敗：\n  " + "\n  ".join(failed))

    analyze(args, classification)


if __name__ == "__main__":
    main()
