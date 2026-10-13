#!/usr/bin/env python3
"""
view_npz.py — 以陣列形式印出 .npz 檔中的所有陣列

用法:
    python view_npz.py data.npz                  # 印出所有陣列(大陣列以 ... 省略中間)
    python view_npz.py data.npz --key depth      # 只印某個陣列
    python view_npz.py data.npz --index 0        # 只印第一軸的第 0 個(例如第 0 個 patch)
    python view_npz.py data.npz --full           # 完整印出,不省略
    python view_npz.py data.npz --precision 2    # 小數位數
需要: numpy
"""
import argparse
import sys

import numpy as np


def main():
    p = argparse.ArgumentParser(description="以陣列形式印出 .npz 檔")
    p.add_argument("path")
    p.add_argument("--key", help="只印這個陣列名稱")
    p.add_argument("--index", type=int, help="只印第一軸的第幾個")
    p.add_argument("--full", action="store_true", help="完整印出,不以 ... 省略")
    p.add_argument("--precision", type=int, default=4, help="小數位數")
    args = p.parse_args()

    np.set_printoptions(precision=args.precision, suppress=True, linewidth=150,
                        threshold=sys.maxsize if args.full else 1000)

    data = np.load(args.path, allow_pickle=True)
    keys = [args.key] if args.key else data.files
    print(f"{args.path}: {len(data.files)} 個陣列")
    for k in data.files:
        a = data[k]
        print(f"  {k:<12} {a.ndim}D  shape={a.shape}  dtype={a.dtype}")
    print()
    for k in keys:
        a = data[k]
        print(f"[{k}] shape={a.shape} dtype={a.dtype}")
        if args.index is not None and a.ndim >= 1:
            a = a[args.index]
            print(f"  取第一軸 index={args.index} -> shape={a.shape}")
        print(a, end="\n\n")


if __name__ == "__main__":
    main()
