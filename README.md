# pi0-FAST on LIBERO-plus：評測與失敗偵測

在 [LIBERO-plus](https://github.com/sylvestf/LIBERO-plus) 擾動基準上評測 [lerobot/pi0fast-libero](https://huggingface.co/lerobot/pi0fast-libero)，並用模型內部特徵訓練失敗偵測器（failure detector）。

## 系統需求

- Linux x86_64（測試環境：Ubuntu 24.04）
- NVIDIA GPU，顯存 ≥ 12GB（測試環境：RTX 3080 Ti）
- NVIDIA 驅動 ≥ 580（PyTorch 2.11 + CUDA 13）
- 磁碟空間約 25GB（conda 環境約 8GB、assets 約 9.5GB、模型權重約 7GB）
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

`setup.sh` 會依序執行：

1. 下載 micromamba 到 `bin/`
2. 在 `env/` 建立 conda 環境（Python 3.12 + ImageMagick 等）
3. clone `lerobot` 和 `LIBERO-plus`，並切換到固定 commit
4. 依 `requirements-lock.txt` 安裝所有 Python 套件（與原實驗版本相同），並以 editable 模式安裝 lerobot、LIBERO-plus
5. 寫入 `~/.libero/config.yaml`（如果原本有這個檔案，會先備份成 `.bak`）
6. 下載並解壓 LIBERO-plus assets 到 `LIBERO-plus/libero/libero/assets`
7. 下載模型權重到 `~/.cache/huggingface`

參數：

| 參數 | 用途 |
|---|---|
| `--nvidia-gl` | 只裝了 NVIDIA 計算驅動、沒有 EGL 的伺服器才需要（見下方說明） |
| `--skip-assets` | 略過 assets 下載 |
| `--skip-model` | 略過模型權重下載 |

腳本可以重複執行，已完成的步驟會自動略過。

### 無螢幕伺服器的 EGL 問題

MuJoCo 用 EGL 在無螢幕環境算圖。如果伺服器只裝了 NVIDIA 計算驅動，執行時出現 EGL 相關錯誤，請改用：

```bash
bash setup.sh --nvidia-gl
```

這個參數會用 `apt download` 下載與目前驅動**同版本**的 `libnvidia-gl-<版本>-server`，解壓到 `nvidia-gl/`（不需要 sudo），`activate.sh` 偵測到這個資料夾就會自動使用。驅動升級後要重新執行一次。

### 驗證安裝

```bash
source activate.sh
python -c "import libero.libero, lerobot, wand.image; print('OK')"
python -c "import mujoco; m=mujoco.MjModel.from_xml_string('<mujoco/>'); mujoco.Renderer(m); print('render OK')"
```

## 使用

每次開新的 terminal 都要先啟用環境：

```bash
source activate.sh
cd eval_pi0fast
```

### 1. 評測

```bash
# 冒煙測試：1 個 suite、每個擾動類別 1 個任務
python eval_pi0fast_libero_plus.py --suites libero_spatial --per-category 1 --chunk-size 7 --output-dir smoke

# 主要實驗：4 個 suite、每類別 20 個任務，並儲存 BDDL 標籤與模型特徵
python eval_pi0fast_libero_plus.py --per-category 20 --bddl-labels --save-features

# 只重新統計已有的結果
python eval_pi0fast_libero_plus.py --analyze-only
```

結果寫入 `eval_pi0fast/outputs/`：`summary.md`（成功率統計）、`results.csv`（逐回合結果）。中斷後用相同參數重跑，會跳過已完成的批次。

### 2. 失敗偵測

```bash
# 單一層：印出各 alpha 與各 suite 的偵測結果
python online_failure_detector.py --feature lm_mean --layer 10 --xlsx online_detection_lm_mean_L10.xlsx

# 逐層掃描並輸出 Excel
python online_failure_detector.py --feature lm_mean --sweep --xlsx layer_sweep_lm_mean.xlsx
```

需要先用 `--bddl-labels --save-features` 跑完評測。

### 3. 查看 npz 特徵檔

```bash
./view_npz.py <檔案>.npz | head -30
```

## 檔案結構

```
.
├── setup.sh                  # 一鍵安裝
├── activate.sh               # 啟用環境（設定 PYTHONPATH、EGL、LD_PRELOAD 等）
├── requirements-lock.txt     # 固定版本的 Python 套件清單
└── eval_pi0fast/
    ├── eval_pi0fast_libero_plus.py   # 評測主程式
    ├── lerobot_eval_bddl.py          # 包裝 lerobot-eval，輸出 BDDL 標籤與特徵
    ├── bddl_labels.py                # 依 BDDL 目標條件產生成功／失敗標籤
    ├── pi0fast_features.py           # 擷取 pi0-FAST 各層特徵
    ├── online_failure_detector.py    # 逐 chunk 的線上失敗偵測
    ├── view_npz.py                   # 印出 npz 檔內容
    └── outputs/                      # 只保留彙整結果；特徵檔與原始輸出不進 git
```

以下內容由 `setup.sh` 產生，不放進 git：`bin/`、`env/`、`micromamba/`、`lerobot/`、`LIBERO-plus/`、`nvidia-gl/`。

## 程式說明

各程式的呼叫關係：

```
eval_pi0fast_libero_plus.py  ──(subprocess 分批呼叫)──▶  lerobot_eval_bddl.py
                                                          ├─ bddl_labels.py      → chunk_*_bddl/*.npz
                                                          └─ pi0fast_features.py → chunk_*_features/*.npz
                                                                     │
                                          online_failure_detector.py ◀──┘
```

### `eval_pi0fast_libero_plus.py`：評測主程式

- 讀取 LIBERO-plus 的 `task_classification.json`，依**擾動類別**分層抽樣任務。LIBERO-plus 有 7 種擾動類別、5 種難度，每個 suite 約 2400 個任務，全部跑完要好幾天，所以預設只抽樣（`--per-category N`），`--all` 才會全部跑。
- 把任務分批（`--chunk-size`），每批用 subprocess 呼叫 `lerobot_eval_bddl.py`，並設定好 `PYTHONPATH`、`LD_PRELOAD`、EGL 等環境變數。每批的輸出放在 `outputs/<suite>/chunk_XXXX/`，log 放在 `chunk_XXXX.log`。
- 中斷後用相同參數重跑，會跳過已完成的批次。
- 全部跑完後彙整結果，輸出 `outputs/results.csv`（每個 episode 一列）和 `outputs/summary.md`（依 suite、擾動類別、難度統計成功率）。加 `--bddl-labels` 時，成功與否以 BDDL 標籤為準。
- 主要參數：`--suites`、`--per-category`、`--all`、`--chunk-size`、`--bddl-labels`、`--save-features`、`--output-dir`、`--analyze-only`。

### `lerobot_eval_bddl.py`：加上記錄功能的 lerobot-eval

用法和 `lerobot-eval` 一樣，但會依環境變數額外記錄資料：

- 有設定 `BDDL_LABEL_DIR`：載入 `bddl_labels.py`，輸出每個 rollout 的成功／失敗標籤
- 有設定 `FEATURE_DIR`：載入 `pi0fast_features.py`，輸出 pi0-FAST 各層特徵

必須搭配 `--eval.batch_size=1`。這兩個環境變數由 `eval_pi0fast_libero_plus.py` 依 `--bddl-labels`、`--save-features` 自動設定，通常不需要直接執行這個檔案。

### `bddl_labels.py`：BDDL 成功標籤

替換 lerobot `LiberoEnv` 的 `reset`、`step`，每一步都用 BDDL 的 `goal_state` 檢查是否所有目標條件都成立。只要任一步全部成立，就記為成功。episode 結束時存成 `<suite>_<task_id>_ep<k>.npz`，內容只有 `label`（`True` = 成功）。它也負責提供目前的 rollout 名稱和步數給 `pi0fast_features.py`。

### `pi0fast_features.py`：擷取模型內部特徵

pi0-FAST 每 10 步呼叫一次 `predict_action_chunk`，這時會對 prefix（3 張影像 × 256 patch + 指令 + BOS）做一次 prefill。這個檔案用 hook 擷取 prefill 階段每一層的輸出，每個 rollout 存成一個 npz（C = chunk 數，約等於步數 / 10）：

| 鍵 | 形狀 | 內容 |
|---|---|---|
| `step` | (C,) | 每個 chunk 開始時的環境步數 |
| `lm_last` | (C, 18, 2048) | Gemma 每層在最後一個 prefix token（BOS）的輸出 |
| `lm_mean` | (C, 18, 2048) | Gemma 每層對所有有效 prefix token 取平均 |
| `vision_mean` | (C, 3, 27, 1152) | SigLIP 每層對 256 個 patch 取平均，3 台相機依序是：主視角、手腕、空白 |

### `online_failure_detector.py`：線上失敗偵測（逐 chunk）

模擬執行過程中即時偵測：每個 chunk 只用「到目前為止」的特徵給出失敗分數，評估能多早發現失敗。

1. **分數**：時間 t 的輸入是前 t+1 個 chunk 特徵的平均（嚴格因果），由 MLP 輸出失敗分數。
2. **門檻**：在訓練折內做 inner CV，以 conformal 分位數決定門檻，使成功 episode 在任何時刻被誤報的比例 ≤ α（`--alphas`，預設 0.05、0.1、0.2）。
3. **偵測**：測試時第一次分數超過門檻的 chunk，就是偵測時間。

模型是標準化 → 一層隱藏層的 MLP。評估使用依「suite × 成功/失敗」分層的 5-fold 交叉驗證，重複 `--repeats` 次取平均。特徵用 `--feature`（`lm_last` / `lm_mean` / `vision_mean`）和 `--layer` 選擇。

輸出偵測率、誤報率、偵測時間（chunk 數，以及佔 timeout 的比例）、早期偵測分數、逐時間 AUROC 等指標。`--sweep` 逐層比較，`--xlsx` 把結果寫成 Excel，例如 `layer_sweep_lm_mean.xlsx`、`online_detection_lm_mean_L10.xlsx`。第一次執行時會建立快取 `outputs/chunk_features_<feature>.npy`。

### `view_npz.py`：查看 npz 內容

印出 `.npz` 裡每個陣列的名稱、形狀、dtype 和數值。可以用 `--key` 只看某個陣列、`--index` 只看第一軸的某一筆、`--full` 完整印出不省略、`--precision` 設定小數位數。

## 已知問題

- **Wand 載入失敗（`jpeg12_*` undefined symbol）**：torchvision 內建的 libjpeg 缺少符號。`activate.sh` 已經用 `LD_PRELOAD` 先載入 conda 的 libjpeg-turbo 來解決，請確認有 `source activate.sh`。
- **`import libero` 抓到原版 LIBERO**：執行 `pip uninstall -y hf-libero`（setup.sh 已經處理）。
- **LIBERO 啟動時詢問 dataset 路徑**：代表 `~/.libero/config.yaml` 不存在，重新執行 `bash setup.sh --skip-assets --skip-model`。
