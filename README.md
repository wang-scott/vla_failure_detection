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
# 以回合平均特徵訓練失敗偵測器
python train_failure_detector.py --feature lm_mean

# 線上（逐 chunk）失敗偵測，掃描各層並輸出 Excel
python online_failure_detector.py --feature lm_mean --sweep --xlsx out.xlsx
```

這兩個腳本需要先用 `--save-features` 跑完評測。

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
    ├── bddl_labels.py
    ├── pi0fast_features.py           # 擷取 pi0-FAST 內部特徵
    ├── train_failure_detector.py
    ├── online_failure_detector.py
    ├── view_npz.py
    └── outputs/                      # 只保留彙整結果；特徵檔與原始輸出不進 git
```

以下內容由 `setup.sh` 產生，不放進 git：`bin/`、`env/`、`micromamba/`、`lerobot/`、`LIBERO-plus/`、`nvidia-gl/`。

## 已知問題

- **Wand 載入失敗（`jpeg12_*` undefined symbol）**：torchvision 內建的 libjpeg 缺少符號。`activate.sh` 已經用 `LD_PRELOAD` 先載入 conda 的 libjpeg-turbo 來解決，請確認有 `source activate.sh`。
- **`import libero` 抓到原版 LIBERO**：執行 `pip uninstall -y hf-libero`（setup.sh 已經處理）。
- **LIBERO 啟動時詢問 dataset 路徑**：代表 `~/.libero/config.yaml` 不存在，重新執行 `bash setup.sh --skip-assets --skip-model`。
