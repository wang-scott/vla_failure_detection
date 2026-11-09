# pi0-FAST on LIBERO-plus：逐層特徵的線上失敗偵測

在 [LIBERO-plus](https://github.com/sylvestf/LIBERO-plus) 擾動基準上評測 [lerobot/pi0fast-libero](https://huggingface.co/lerobot/pi0fast-libero)，
擷取模型**每一層**在讀入觀測（prefill）與產生動作（解碼）時的特徵，比較哪一層、哪一種特徵最能在執行過程中**及早且可靠地**偵測失敗。

實驗重點：

- **解碼階段特徵 `act_mean`**：模型產生動作 token 時的 hidden state（同 SAFE 論文的做法），與 prefill 的 `lm_mean`（影像 + 指令）比較。
- **不依賴初始圖像與語言**：每個 chunk 的特徵都含有場景與指令資訊，只看第一個 chunk 就能猜出「這個任務容易失敗」。
  預設以 `e_t − e_0` 去掉這部分（`--input sub_first`），讓偵測只能來自執行過程中的變化。
- **排除「靠長度偵測」**：失敗 episode 都跑到 timeout，因此只計入偵測期限（同任務最短成功 episode 長度）內的警報，並與「只看時間 t」的長度基準比較。

## 系統需求

- Linux x86_64（測試環境：Ubuntu 24.04）
- NVIDIA GPU，顯存 ≥ 12GB（測試環境：RTX 3080 Ti）
- NVIDIA 驅動 ≥ 580（PyTorch 2.11 + CUDA 13）
- 磁碟空間：安裝約 25GB（conda 環境約 8GB、assets 約 9.5GB、模型權重約 7GB），另需評測輸出與特徵的空間（舊版只存 prefill 特徵時約 6GB，加入解碼特徵後會更多）
- `git`、`curl`、`wget`、`unzip`
- Hugging Face 帳號，並已在 [google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224) 頁面同意授權（pi0-FAST 用它的 tokenizer）

## 安裝

```bash
git clone <本 repo 的網址> vla
cd vla
# 先登入 Hugging Face（下載 paligemma tokenizer 需要）
#   還沒有 hf 指令的話，可以先跑完 setup.sh，再執行 source activate.sh && hf auth login
bash setup.sh
```

`setup.sh` 依序：下載 micromamba → 建立 `env/` → clone 固定 commit 的 `lerobot`、`LIBERO-plus` →
依 `requirements-lock.txt`（實驗環境的 pip freeze）安裝套件 → 寫入 `~/.libero/config.yaml` →
下載 LIBERO-plus assets 與模型權重 → 設定 EGL 並檢查環境（套件可 import、EGL 看得到 GPU、robosuite 能算圖）。
可重複執行，已完成的步驟會略過。

| 參數 | 用途 |
|---|---|
| `--skip-assets` | 略過 assets 下載 |
| `--skip-model` | 略過模型權重下載 |
| `--nvidia-gl` | 強制下載與驅動同版本的 libnvidia-gl 到 `nvidia-gl/`（通常不需要，會自動判斷） |

**EGL**：MuJoCo 在無螢幕環境用 EGL 算圖，EGL 函式庫版本必須與**目前載入的**驅動相同。驅動升級並重開機後
（出現 `Cannot initialize a EGL device display`），開新的 terminal 執行 `bash setup.sh --skip-assets --skip-model`。
已經 `source activate.sh` 過的 terminal 會留著舊的 EGL 環境變數，修好後要開新的 terminal。

## 執行實驗

完整指令也整理在 [`rerun_commands.txt`](rerun_commands.txt)。每次開新的 terminal：

```bash
source activate.sh
cd eval_pi0fast
export OUTPUTS_DIR=outputs_act      # 分析程式讀取的評測輸出資料夾
```

### 1. 評測並擷取特徵

```bash
# 冒煙測試（7 個任務）：確認模擬與特徵擷取正常
python eval_pi0fast_libero_plus.py --suites libero_spatial --per-category 1 --chunk-size 7 \
    --bddl-labels --save-features --output-dir smoke

# 主要實驗：4 個 suite × 7 種擾動類別 × 20 個任務（共 560 個 episode），時間很長，建議在 tmux 中執行
python eval_pi0fast_libero_plus.py --per-category 20 --bddl-labels --save-features --output-dir outputs_act

# 只重新統計結果
python eval_pi0fast_libero_plus.py --analyze-only --output-dir outputs_act
```

中斷後以相同指令重跑會跳過已完成的批次。輸出：`outputs_act/results.csv`（逐 episode）、`summary.md`（成功率統計）、
`<suite>/chunk_*_features/*.npz`（特徵）、`<suite>/chunk_*_bddl/*.npz`（成功標籤）、`<suite>/chunk_*/videos/`（影片）。

### 2. 逐層比較

```bash
# 解碼階段特徵（18 層）與 prefill 特徵（18 層）
python online_failure_detector.py --feature act_mean --sweep --repeats 1 --xlsx sweep_act_mean.xlsx
python online_failure_detector.py --feature lm_mean  --sweep --repeats 1 --xlsx sweep_lm_mean.xlsx
# 論文使用的 pre-logits（最後一層經 final RMSNorm，只有 1 層）
python online_failure_detector.py --feature act_prelogits --layer 0

# 換成 prefix 分數
python online_failure_detector.py --feature act_mean --sweep --repeats 1 --score prefix --xlsx sweep_act_mean_prefix.xlsx

# 未見任務：測試的基礎任務在訓練時沒看過，確認不是靠記住任務
python online_failure_detector.py --feature act_mean --sweep --repeats 1 --split task --xlsx sweep_act_mean_task.xlsx

# 選定層後以 3 次重複確認
python online_failure_detector.py --feature act_mean --layer <L> --repeats 3
```

**判讀時請一併檢查**，確認偵測不是來自初始特徵或長度：

| 檢查 | 做法 | 期望 |
|---|---|---|
| 只看開頭能偵測多少 | `--input first` | AUROC 明顯低於正常設定 |
| 一開始是否就分得出來 | 報表的「逐時間 AUROC」t=1、2 | 接近 0.5，之後才上升 |
| 是否勝過長度基準 | 摘要表的「長度基準」列 | BA、早期偵測分數明顯較高 |
| 未見任務 | `--split task` | 與 `--split episode` 差距不大 |

### 3. 對照影片人工檢查

```bash
python export_episode_scores.py --feature act_mean --layer <L> --out outputs_act/episode_scores_act_mean_L<L>
python render_detection_videos.py --scores outputs_act/episode_scores_act_mean_L<L>.csv
# 與另一個設定的同一批 episode 對照：--same_as <另一個資料夾>/index.csv
```

影片輸出在 `outputs_act/annotated_videos/<CSV 檔名>/`，自動挑選最早被偵測的失敗（`fail_early_*`）、
被誤報的成功（`false_alarm_*`）、未偵測的失敗等，清單在 `index.csv`。畫面下方的面板：

- 白線：失敗分數 s（只畫到目前的 chunk）
- 黃虛線：門檻 τ（由訓練集成功 episode 校準，與目前這個 episode 無關）
- 紅線：警報時間（s 第一次超過 τ）；灰線：偵測期限

## 特徵

pi0-FAST 每 10 個環境步呼叫一次 `predict_action_chunk`：先對 prefix（3 張影像 × 256 patch + 指令 + BOS）做一次
prefill，再逐 token 解碼 FAST 動作 token（KV cache，greedy）直到結束符號 `|`。Gemma 共 18 層，每一層都會對每個位置
產生 hidden state；只有最後一層的輸出經 final RMSNorm → lm_head 決定下一個動作 token。

```
[ 影像 patch | 指令 | BOS ] → 動作 token 1 → 動作 token 2 → … → "|"
└──────── prefill ────────┘ └──────────────── 解碼 ────────────────┘
 lm_mean = 這些位置的平均         act_mean = 產生每個動作 token 的位置的平均
```

每個 rollout 存成 `<suite>_<task_id>_ep<k>.npz`（C = chunk 數）：

| 鍵 | 形狀 | 內容 |
|---|---|---|
| `step` | (C,) | 每個 chunk 開始時的環境步數 |
| `act_mean` | (C, 18, 2048) | 解碼：每層在產生動作 token 的位置取平均（SAFE 的 aggtoken=Mean, encoded） |
| `act_prelogits` | (C, 1, 2048) | 同上，最後一層經 final RMSNorm 之後（SAFE 的 pre-logits） |
| `n_act_tokens` | (C,) | 每個 chunk 產生的動作 token 數（含結束符號） |
| `lm_mean` | (C, 18, 2048) | prefill：每層對所有有效 prefix token 取平均（以影像 patch 為主） |
| `lm_last` | (C, 18, 2048) | prefill：每層在 BOS 位置的輸出 |
| `vision_mean` | (C, 3, 27, 1152) | SigLIP 每層對 256 個 patch 取平均（主視角、手腕、空白相機） |

`lm_mean` 以影像 patch 為主，強烈反映場景與指令（只看第一個 chunk 就有 AUROC 約 0.83）；
`act_mean` 是模型產生動作時的表示，預期更接近執行狀況。

## 失敗偵測（`online_failure_detector.py`）

每個 chunk 只用「到目前為止」的特徵給出失敗分數 s_t，第一次超過門檻 τ(t) 時發出警報。

**預設設定**

| 參數 | 預設 | 說明 |
|---|---|---|
| `--feature` / `--layer` | `lm_mean` / 12 | 特徵與層；新實驗請指定 `--feature act_mean` |
| `--input` | `sub_first` | `e_t − e_0`，去掉初始圖像 / 語言；另有 `raw`、`delta`（`e_t − e_{t−1}`）、`first`（診斷用，只給 `e_0`） |
| `--skip` | 0 | 略過前 k 個 chunk；`sub_first` / `delta` 時第 0 個 chunk 恆為 0、不含資訊，自動略過 |
| `--score` | `cumsum` | `cumsum`：s_t = Σ sigmoid(g(e_τ))（同 SAFE-MLP）；`prefix`：前綴平均特徵 → MLP logit（可用 `--window` 只取最近幾個 chunk） |
| `--threshold` | cumsum 為 `functional`，prefix 為 `const` | `functional`：τ(t) = μ(t) + h·m(t)（functional CP，同 SAFE）；`const`：固定門檻 |
| `--alphas` / `--alpha` | 0.05,0.1,0.2 / 0.1 | 成功 episode 被誤報比例的上限 |
| `--deadline` | `min` | 只計入同基礎任務最短成功 episode 長度內的警報（同 SAFE 的 by earliest stop） |
| `--split` | `episode` | `episode`：看過的任務；`task`：以基礎任務切分，未見任務 |
| `--cumsum_bce` | 0 | cumsum 另加逐 chunk BCE，減少 sigmoid 飽和 |

門檻在訓練折內以 inner CV 取得成功 episode 的分數曲線後用 conformal 方法校準；5-fold 交叉驗證，重複 `--repeats` 次取平均。
LIBERO-plus 的任務是「原始 LIBERO 任務 + 一種擾動」，所有 episode 來自 40 個基礎任務（每個 suite 10 個），基礎任務由任務名稱去掉擾動後綴得到。

**輸出指標**：失敗偵測率（TPR）、成功誤報率（FPR）、Balanced Accuracy、偵測時間 T_det（chunk 數與佔 timeout 比例）、
早期偵測分數（失敗 episode 平均剩餘時間比例，未偵測算 0）、episode 層級 AUROC、逐時間 AUROC，以及同樣條件下的長度基準。
`--sweep` 逐層比較，`--xlsx` 輸出 Excel。第一次執行會在 `$OUTPUTS_DIR` 建立特徵快取 `chunk_features_<feature>.npy`。

**已知限制（cumsum）**：分數每個 chunk 最多加 1，若超過 α 比例的成功 episode 也被判為 p≈1，門檻會貼近上限 t，
要等成功 episode 結束、門檻變平後才可能警報，結果接近長度基準。此時可比較 `--score prefix`。

## 檔案結構

```
.
├── setup.sh                  # 一鍵安裝與環境檢查
├── activate.sh               # 啟用環境（PYTHONPATH、EGL、LD_PRELOAD 等）
├── requirements-lock.txt     # 實驗環境的 pip freeze
├── rerun_commands.txt        # 本實驗的完整指令
└── eval_pi0fast/
    ├── eval_pi0fast_libero_plus.py   # 評測主程式（分批、可續跑、彙整成功率）
    ├── lerobot_eval_bddl.py          # 包裝 lerobot-eval，輸出 BDDL 標籤與特徵
    ├── bddl_labels.py                # 依 BDDL 目標條件產生成功／失敗標籤
    ├── pi0fast_features.py           # 擷取 prefill 與解碼階段的各層特徵
    ├── online_failure_detector.py    # 逐 chunk 的線上失敗偵測與逐層比較
    ├── export_episode_scores.py      # 輸出逐 episode 的分數、門檻、偵測時間（CSV / NPZ）
    ├── render_detection_videos.py    # 把分數曲線與警報疊到評測影片上
    ├── annotate_chunks.py            # 在影片上標記 chunk 編號
    └── view_npz.py                   # 印出 npz 內容
```

呼叫關係：

```
eval_pi0fast_libero_plus.py ──(subprocess 分批)──▶ lerobot_eval_bddl.py
                                                    ├─ bddl_labels.py      → chunk_*_bddl/*.npz
                                                    └─ pi0fast_features.py → chunk_*_features/*.npz
online_failure_detector.py ◀── 特徵與標籤（$OUTPUTS_DIR）
export_episode_scores.py ──▶ render_detection_videos.py ──▶ annotated_videos/
```

`bin/`、`env/`、`micromamba/`、`lerobot/`、`LIBERO-plus/`、`nvidia-gl/` 由 `setup.sh` 產生，不放進 git。

## 已知問題

- **EGL 找不到 GPU（`Cannot initialize a EGL device display`）**：見「安裝」的 EGL 說明。
- **Wand 載入失敗（`jpeg12_*` undefined symbol）**：`activate.sh` 以 `LD_PRELOAD` 先載入 conda 的 libjpeg-turbo，請確認有 `source activate.sh`。
- **Sensor Noise 擾動出現 `np.float_ was removed in the NumPy 2.0 release`**：LIBERO-plus 使用了 NumPy 2 已移除的別名，而 lerobot 需要 numpy>=2，無法降級；`setup.sh` 會把它改成等價的 `np.float64`。同一擾動的 fog 固定產生 256×256 的雜訊圖，與 lerobot 的 360×360 影像不合（`operands could not be broadcast`），`setup.sh` 也會一併修正。舊的 `outputs/` 因此缺少 Sensor Noise（只有 480 個 episode）。
- **`import libero` 抓到原版 LIBERO**：執行 `pip uninstall -y hf-libero`（setup.sh 已處理）。
- **LIBERO 啟動時詢問 dataset 路徑**：`~/.libero/config.yaml` 不存在，執行 `bash setup.sh --skip-assets --skip-model`。
- **解碼特徵需要 `use_kv_cache=True`**（預設）；`pi0fast_features.py` 偵測到關閉時會直接報錯。
