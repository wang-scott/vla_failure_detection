"""與 lerobot-eval 相同，但可額外輸出每個 rollout 的 BDDL 標籤與 pi0-FAST 各層特徵。

BDDL_LABEL_DIR 有設定 -> 輸出成功/失敗標籤（bddl_labels.py）
FEATURE_DIR 有設定    -> 輸出各層特徵（pi0fast_features.py）
需搭配 --eval.batch_size=1（同步環境），記錄器才會在同一個行程內生效。
"""

import os
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from bddl_labels import install_recorder  # noqa: E402

install_recorder()

if "FEATURE_DIR" in os.environ:
    from pi0fast_features import install_feature_recorder  # noqa: E402

    install_feature_recorder()

from lerobot.scripts.lerobot_eval import main  # noqa: E402

if __name__ == "__main__":
    main()
