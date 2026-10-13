"""依 BDDL 目標條件，為每筆 rollout 產生成功／失敗標籤。

install_recorder() 會替換 lerobot 的 LiberoEnv.reset/step，每一步以 BDDL
goal_state 檢查是否所有目標條件都成立；episode 結束時存成
<BDDL_LABEL_DIR>/<suite>_<task_id>_ep<k>.npz，內容只有：

    label: bool, shape (1,)   True = 成功（任一步所有目標條件成立）, False = 失敗
"""

import os
from pathlib import Path

import numpy as np

# 目前 rollout 的名稱與步數，供 pi0fast_features 對應特徵檔使用
CURRENT = {"key": None, "step": 0}


def _flush(libero_env):
    if not getattr(libero_env, "_bddl_active", False) or "BDDL_LABEL_DIR" not in os.environ:
        return
    out_dir = Path(os.environ["BDDL_LABEL_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _key(libero_env)
    np.savez_compressed(out_dir / f"{stem}.npz", label=np.array([libero_env._bddl_success], dtype=bool))
    libero_env._bddl_active = False


def _key(libero_env) -> str:
    return f"{libero_env._bddl_suite_name}_{libero_env.task_id}_ep{libero_env._bddl_episode}"


def install_recorder():
    """替換 lerobot LiberoEnv 的 reset/step/close，必須在建立環境前呼叫。"""
    from lerobot.envs import libero as lr_libero

    cls = lr_libero.LiberoEnv
    if getattr(cls, "_bddl_patched", False):
        return
    orig_init, orig_reset, orig_step, orig_close = cls.__init__, cls.reset, cls.step, cls.close

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self._bddl_suite_name = kwargs.get("task_suite_name", args[2] if len(args) > 2 else "unknown")
        self._bddl_episode = -1
        self._bddl_active = False

    def reset(self, *args, **kwargs):
        _flush(self)
        result = orig_reset(self, *args, **kwargs)
        self._bddl_episode += 1
        self._bddl_success = False
        self._bddl_active = True
        CURRENT.update(key=_key(self), step=0)
        return result

    def step(self, action):
        result = orig_step(self, action)
        CURRENT["step"] += 1
        if self._bddl_active:
            self._bddl_success |= bool(self._env.env._check_success())  # BDDL goal_state 全部成立
        return result

    def close(self):
        _flush(self)
        return orig_close(self)

    cls.__init__, cls.reset, cls.step, cls.close = __init__, reset, step, close
    cls._bddl_patched = True
