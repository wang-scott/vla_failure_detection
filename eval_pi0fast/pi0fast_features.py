"""在評測時擷取 pi0-FAST 每一層的特徵。

pi0-FAST 每 n_action_steps（=10）步呼叫一次 predict_action_chunk：
先對 prefix [3 張影像 x 256 patch, 指令 tokens, BOS] 做一次 prefill，
再逐 token 解碼 FAST 動作（KV cache，每步只送入 1 個 token）。prefill 與解碼階段的特徵都會擷取。

每個 rollout 存成 <FEATURE_DIR>/<suite>_<task_id>_ep<k>.npz（與 BDDL 標籤同名），
C = 該 rollout 的 chunk 數（約 步數 / 10）：

    step          (C,)                  每個 chunk 開始時的環境步數
    lm_last       (C, 18, 2048) float32 Gemma 每層輸出在最後一個 prefix token（BOS）的向量，
                                        也就是用來預測第一個動作 token 的位置
    lm_mean       (C, 18, 2048) float32 Gemma 每層輸出對所有有效 prefix token 取平均
    vision_mean   (C, 3, 27, 1152) float32  SigLIP 每層輸出對 256 個 patch 取平均，
                                        相機順序：image（主視角）, image2（手腕）, empty_camera_0（空白）
    act_mean      (C, 18, 2048) float32 解碼階段：Gemma 每層輸出在「產生每個動作 token 的位置」取平均
                                        （prefill 的 BOS 位置產生第 1 個 token，之後每個解碼步產生下一個），
                                        即 SAFE 論文的 aggtoken=Mean（encoded，final RMSNorm 之前）
    act_prelogits (C, 1, 2048)  float32 同上，但取 final RMSNorm 之後、lm_head 之前（SAFE 的 pre-logits + Mean）
    n_act_tokens  (C,)                  每個 chunk 產生的動作 token 數（含結束符號 "|"）

以 float32 儲存：Gemma 較深層的 residual stream 數值很大（act_mean 最後一層可超過 65504），
float16 會溢位成 inf。

需搭配 bddl_labels.install_recorder()（提供目前的 rollout 名稱與步數），
並以 --eval.batch_size=1 執行（假設 batch 大小為 1）；解碼特徵需要 use_kv_cache=True（預設）。
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
        pg.model.language_model.norm.register_forward_hook(self._norm_hook)
        if not getattr(policy.config, "use_kv_cache", True):
            raise RuntimeError("解碼特徵假設 use_kv_cache=True（每個解碼步只送入 1 個 token）")
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

    def _add_act(self, key, i, v):
        """累加產生動作 token 的位置的向量（留在 GPU 上，chunk 結束才搬回 CPU）。"""
        acc = self.cur[key]
        acc[i] = v if acc[i] is None else acc[i] + v

    def _lm_hook(self, i):
        def hook(module, args, output):
            h = self._tensor(output)
            if self.cur is None:
                return
            h = h[0].float()
            self._add_act("act_sum", i, h[-1])  # prefill 的 BOS 位置或解碼步的唯一位置，都會產生下一個動作 token
            if h.shape[0] == 1:  # 解碼步
                return
            mask = self.pad_mask[0].to(h.device) if self.pad_mask is not None else torch.ones(h.shape[0], dtype=torch.bool, device=h.device)
            self.cur["lm_last"][i] = h[-1].cpu().numpy()
            self.cur["lm_mean"][i] = h[mask].mean(0).cpu().numpy()

        return hook

    def _norm_hook(self, module, args, output):
        """final RMSNorm 的輸出（lm_head 的輸入）；每次 Gemma forward（prefill 或解碼步）呼叫一次。"""
        if self.cur is None:
            return
        self._add_act("prelogit_sum", 0, self._tensor(output)[0, -1].float())
        self.cur["n_act"] += 1

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
            "act_sum": [None] * self.n_lm,
            "prelogit_sum": [None],
            "n_act": 0,
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
                "act_mean": torch.stack(c["act_sum"]).div(c["n_act"]).cpu().numpy(),
                "act_prelogits": torch.stack(c["prelogit_sum"]).div(c["n_act"]).cpu().numpy(),
                "n_act_tokens": c["n_act"],
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
            lm_last=np.stack([c["lm_last"] for c in self.chunks]).astype(np.float32),
            lm_mean=np.stack([c["lm_mean"] for c in self.chunks]).astype(np.float32),
            vision_mean=np.stack([c["vision_mean"] for c in self.chunks]).astype(np.float32),
            act_mean=np.stack([c["act_mean"] for c in self.chunks]).astype(np.float32),
            act_prelogits=np.stack([c["act_prelogits"] for c in self.chunks]).astype(np.float32),
            n_act_tokens=np.array([c["n_act_tokens"] for c in self.chunks], dtype=np.int32),
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
