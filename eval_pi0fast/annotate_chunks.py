"""在評估影片上標記每一幀屬於第幾個 action chunk（不修改原始影片）。

影片為每個環境步一幀；pi0-FAST 每 n_action_steps 步呼叫一次 predict_action_chunk，
各 chunk 開始時的環境步數記錄在 chunk_*_features/*.npz 的 step 中，第 c 個 chunk 涵蓋第 step[c] ~ step[c+1]-1 幀。

用法：
    python annotate_chunks.py --suite libero_10 --task_id 49 --episode 0
    python annotate_chunks.py --suite libero_10 --task_id 49 --episode 0 --fps 10
"""

import argparse
import glob
import os
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

# 評測輸出資料夾；可用環境變數 OUTPUTS_DIR 指定（例如重跑評估後的 outputs_act）
ROOT = Path(os.environ.get("OUTPUTS_DIR", Path(__file__).parent / "outputs"))


def find_one(pattern):
    files = glob.glob(str(ROOT / pattern))
    if len(files) != 1:
        raise FileNotFoundError(f"{pattern} 找到 {len(files)} 個檔案：{files}")
    return Path(files[0])


def annotate(frames, step):
    """frames (T, H, W, 3) RGB、step (C,) 各 chunk 開始的環境步數 -> 左上角標上 chunk 編號的影格。"""
    chunk = np.searchsorted(step, np.arange(len(frames)), side="right") - 1
    out = []
    for i, f in enumerate(frames):
        f = f.copy()
        scale = f.shape[0] / 360
        lines = [f"chunk {chunk[i]} / {len(step) - 1}", f"step {i}"]
        for j, text in enumerate(lines):
            org = (int(8 * scale), int((24 + 24 * j) * scale))
            cv2.putText(f, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (0, 0, 0), int(4 * scale), cv2.LINE_AA)
            cv2.putText(f, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (255, 255, 0), int(1.5 * scale), cv2.LINE_AA)
        out.append(f)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", required=True)
    p.add_argument("--task_id", type=int, required=True)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--fps", type=float, default=20, help="輸出影片的 fps（原始影片為 80，太快不易觀察）")
    p.add_argument("--out_dir", type=str, default=str(ROOT / "annotated_videos"))
    args = p.parse_args()

    name = f"{args.suite}_{args.task_id}"
    video = find_one(f"{args.suite}/chunk_*/videos/{name}/eval_episode_{args.episode}.mp4")
    step = np.load(find_one(f"{args.suite}/chunk_*_features/{name}_ep{args.episode}.npz"))["step"]
    frames = imageio.mimread(video, memtest=False)
    print(f"{video}：{len(frames)} 幀，{len(step)} 個 chunk（step {step[0]} ~ {step[-1]}）")
    if len(frames) < step[-1] + 1:
        print(f"警告：影片幀數少於最後一個 chunk 的起始步數 {step[-1]}，幀與步數可能沒有一一對應")

    out = Path(args.out_dir) / f"{name}_ep{args.episode}_chunks.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out, annotate(frames, step), fps=args.fps, macro_block_size=8)
    print(f"已寫出 {out}")


if __name__ == "__main__":
    main()
