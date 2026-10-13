"""在評測時擷取 pi0-FAST 每一層的特徵。

pi0-FAST 每 n_action_steps（=10）步呼叫一次 predict_action_chunk：
先對 prefix [3 張影像 x 256 patch, 指令 tokens, BOS] 做一次 prefill，
再逐 token 解碼 FAST 動作。這裡只擷取 prefill 的特徵（解碼階段每次只有 1 個 token）。

每個 rollout 存成 <FEATURE_DIR>/<suite>_<task_id>_ep<k>.npz（與 BDDL 標籤同名），
C = 該 rollout 的 chunk 數（約 步數 / 10）：

    step          (C,)                  每個 chunk 開始時的環境步數
    lm_last       (C, 18, 2048) float16 Gemma 每層輸出在最後一個 prefix token（BOS）的向量，
                                        也就是用來預測第一個動作 token 的位置
    lm_mean       (C, 18, 2048) float16 Gemma 每層輸出對所有有效 prefix token 取平均
    vision_mean   (C, 3, 27, 1152) float16  SigLIP 每層輸出對 256 個 patch 取平均，
                                        相機順序：image（主視角）, image2（手腕）, empty_camera_0（空白）

需搭配 bddl_labels.install_recorder()（提供目前的 rollout 名稱與步數），
並以 --eval.batch_size=1 執行（假設 batch 大小為 1）。
"""

import atexit
import os
from pathlib import Path

import numpy as np
import torch

import bddl_labels


class _Recorder:
    def __init__(self, policy):
        self.key = None
        self.chunks = []
        self.cur = None  # 目前 chunk 的暫存
        self.pad_mask = None
        pg = policy.model.paligemma_with_expert.paligemma
        lm_layers = pg.model.language_model.layers
        vis_layers = pg.model.vision_tower.vision_model.encoder.layers
        self.n_lm, self.n_vis = len(lm_layers), len(vis_layers)
        for i, layer in enumerate(lm_layers):
            layer.register_forward_hook(self._lm_hook(i))
        for i, layer in enumerate(vis_layers):
            layer.register_forward_hook(self._vis_hook(i))
        # 取得 prefix 的有效 token mask（取平均時排除 padding 與被遮蔽的空白相機）
        model = policy.model
        orig_embed = model.embed_prefix_fast

        def embed_prefix_fast(*args, **kwargs):
            out = orig_embed(*args, **kwargs)
            self.pad_mask = out[1]
            return out

        model.embed_prefix_fast = embed_prefix_fast

    @staticmethod
    def _tensor(output):
        return output[0] if isinstance(output, tuple) else output

    def _lm_hook(self, i):
        def hook(module, args, output):
            h = self._tensor(output)
            if self.cur is None or h.shape[1] == 1:  # 只取 prefill，略過逐 token 解碼
                return
            h = h[0].float()
            mask = self.pad_mask[0].to(h.device) if self.pad_mask is not None else torch.ones(h.shape[0], dtype=torch.bool, device=h.device)
            self.cur["lm_last"][i] = h[-1].cpu().numpy()
            self.cur["lm_mean"][i] = h[mask].mean(0).cpu().numpy()

        return hook

    def _vis_hook(self, i):
        def hook(module, args, output):
            if self.cur is None:
                return
            h = self._tensor(output)[0].float()  # (256, 1152)
            cam = self.cur["vis_calls"] // self.n_vis  # 每張影像依序跑完 27 層
            self.cur["vision_mean"].setdefault(cam, [None] * self.n_vis)[i] = h.mean(0).cpu().numpy()
            self.cur["vis_calls"] += 1

        return hook

    def begin_chunk(self):
        key = bddl_labels.CURRENT["key"]
        if key != self.key:
            self.flush()
            self.key = key
        self.cur = {
            "step": bddl_labels.CURRENT["step"],
            "lm_last": [None] * self.n_lm,
            "lm_mean": [None] * self.n_lm,
            "vision_mean": {},
            "vis_calls": 0,
        }

    def end_chunk(self):
        c = self.cur
        self.cur = None
        cams = [np.stack(c["vision_mean"][k]) for k in sorted(c["vision_mean"])]
        self.chunks.append(
            {
                "step": c["step"],
                "lm_last": np.stack(c["lm_last"]),
                "lm_mean": np.stack(c["lm_mean"]),
                "vision_mean": np.stack(cams),
            }
        )

    def flush(self):
        if self.key is None or not self.chunks:
            self.chunks = []
            return
        out_dir = Path(os.environ["FEATURE_DIR"])
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / f"{self.key}.npz",
            step=np.array([c["step"] for c in self.chunks], dtype=np.int32),
            lm_last=np.stack([c["lm_last"] for c in self.chunks]).astype(np.float16),
            lm_mean=np.stack([c["lm_mean"] for c in self.chunks]).astype(np.float16),
            vision_mean=np.stack([c["vision_mean"] for c in self.chunks]).astype(np.float16),
        )
        self.chunks = []


def install_feature_recorder():
    """替換 PI0FastPolicy.predict_action_chunk，第一次呼叫時掛上各層 hook。"""
    from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy

    if getattr(PI0FastPolicy, "_feature_patched", False):
        return
    orig_predict = PI0FastPolicy.predict_action_chunk

    def predict_action_chunk(self, batch, **kwargs):
        if not hasattr(self, "_feature_recorder"):
            self._feature_recorder = _Recorder(self)
            atexit.register(self._feature_recorder.flush)
        rec = self._feature_recorder
        rec.begin_chunk()
        try:
            return orig_predict(self, batch, **kwargs)
        finally:
            rec.end_chunk()

    PI0FastPolicy.predict_action_chunk = predict_action_chunk
    PI0FastPolicy._feature_patched = True
