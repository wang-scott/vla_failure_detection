# agent.md：專案結構與程式功能

給接手此專案的人或 AI agent 的快速導覽。實驗背景見 [experiment_report.md](experiment_report.md)，指令見 [rerun_commands.txt](rerun_commands.txt)，安裝與使用見 [README.md](README.md)。

## 專案在做什麼

在 LIBERO-plus 擾動基準上評測 π0-FAST（`lerobot/pi0fast-libero`）。評測時擷取模型每一層在 prefill（讀入影像與指令）與解碼（產生動作 token）時的特徵，再做逐 chunk 的線上失敗偵測（沿用 SAFE 的框架），逐層比較哪種特徵最能及早偵測失敗。偵測時會刻意避免依賴初始圖像與語言，以及依賴 episode 長度。

## 資料夾結構

```
vla/
├── setup.sh                  # 一鍵安裝：micromamba、conda env、lerobot / LIBERO-plus（固定 commit）、套件、assets、模型、EGL 檢查
├── activate.sh               # 每個新 terminal 都要 source：conda env、PYTHONPATH、LD_PRELOAD、MUJOCO_GL=egl、EGL 設定
├── requirements-lock.txt     # 實驗環境的 pip freeze（numpy 2.2.6；lerobot 要求 numpy>=2,<2.3）
├── README.md                 # 安裝與執行說明（只描述目前的新實驗）
├── experiment_report.md      # 實驗設計報告（給熟悉 SAFE 論文的讀者）
├── rerun_commands.txt        # 重跑評估與分析的完整指令
├── command.txt               # 舊實驗的指令紀錄
├── 2506.09937v2.pdf          # SAFE 論文
├── eval_pi0fast/             # 所有實驗程式與輸出（見下方）
│
│   以下由 setup.sh 產生，不進 git
├── bin/  micromamba/  env/   # micromamba 與 conda 環境（Python 3.12）
├── lerobot/                  # huggingface/lerobot @ ff71cae（editable 安裝）
├── LIBERO-plus/              # sylvestf/LIBERO-plus @ 4976dc3（editable 安裝；setup.sh 會修正 np.float_）
└── nvidia-gl.old-595.71.05/  # 舊驅動版本的 libnvidia-gl，已停用（系統 EGL 與目前驅動相符）
```

### `eval_pi0fast/`

```
eval_pi0fast/
├── eval_pi0fast_libero_plus.py   # 評測主程式
├── lerobot_eval_bddl.py          # lerobot-eval 包裝：掛上標籤與特徵記錄器
├── bddl_labels.py                # BDDL 成功標籤 + 提供目前 rollout 名稱與步數
├── pi0fast_features.py           # 擷取各層特徵（prefill + 解碼）
├── online_failure_detector.py    # 線上失敗偵測、逐層比較
├── export_episode_scores.py      # 輸出逐 episode 的分數、門檻、偵測時間
├── render_detection_videos.py    # 把分數曲線與警報疊到評測影片上
├── annotate_chunks.py            # 在影片上標 chunk 編號（render 會用到它的 ROOT / find_one）
├── view_npz.py                   # 印出 npz 內容
├── outputs/                      # 舊實驗：480 episode，只有 prefill 特徵，缺 Sensor Noise
└── outputs_act/                  # 新實驗：560 episode，含解碼特徵（評測進行中）
```

## 程式功能與資料流

```
eval_pi0fast_libero_plus.py ──(subprocess，每批 20 個任務)──▶ lerobot_eval_bddl.py
                                                               ├─ bddl_labels.py      → <suite>/chunk_XXXX_bddl/*.npz
                                                               └─ pi0fast_features.py → <suite>/chunk_XXXX_features/*.npz
                              彙整 ──▶ results.csv、summary.md

online_failure_detector.py  ◀── 特徵 + 標籤 + results.csv（由 $OUTPUTS_DIR 指定資料夾）
export_episode_scores.py    ──（呼叫上者）──▶ episode_scores_*.csv / .npz
render_detection_videos.py  ◀── episode_scores_*.csv + 原始影片 ──▶ annotated_videos/<名稱>/
```

| 程式 | 功能 | 重點 |
|---|---|---|
| `eval_pi0fast_libero_plus.py` | 依擾動類別分層抽樣任務（`--per-category`，固定 `--seed`），分批呼叫 lerobot-eval，最後彙整成功率 | 抽樣計畫存在 `plan.json`，參數不同會拒絕寫入同一個資料夾；已完成的批次（有 `eval_info.json`）會跳過，可中斷續跑；`--bddl-labels`、`--save-features` 會設定 `BDDL_LABEL_DIR`、`FEATURE_DIR` |
| `lerobot_eval_bddl.py` | 與 `lerobot-eval` 相同，依環境變數載入記錄器 | 必須用 `--eval.batch_size=1` |
| `bddl_labels.py` | 每一步檢查 BDDL 目標條件，任一步全部成立即為成功，輸出 `label` | 同時維護 `CURRENT = {key, step}`，供特徵記錄器命名與對齊 |
| `pi0fast_features.py` | 以 forward hook 擷取每個 chunk 的特徵：`lm_mean`、`lm_last`（prefill）、`act_mean`、`act_prelogits`、`n_act_tokens`（解碼）、`vision_mean`（SigLIP）、`step` | 解碼特徵假設 KV cache（每個解碼步只有 1 個 token）；`act_*` 的平均對象是「產生每個動作 token 的位置」（BOS + 各解碼步） |
| `online_failure_detector.py` | 交叉驗證訓練 MLP 失敗分數，用 conformal 方法校準門檻，計算 TPR、FPR、BA、T_det、早期偵測分數、AUROC，並附長度基準 | 預設：`--input sub_first`（e_t − e_0，自動略過 chunk 0）、`--score cumsum`、門檻 functional、`--deadline min`、`--feature lm_mean --layer 12`；`--sweep` 逐層比較，`--xlsx` 輸出 Excel；第一次執行會在 `$OUTPUTS_DIR` 建立快取 `chunk_features_<feature>.npy` |
| `export_episode_scores.py` | 執行上者（參數原樣傳入），攔截結果，輸出每個 episode 的偵測時間與逐 chunk 分數、門檻 | `alarm_chunk` = 影片上標示的 chunk 編號 |
| `render_detection_videos.py` | 自動挑選 episode（`fail_early`、`false_alarm`、`fail_late`、`fail_missed`、`success_ok`），輸出帶分數面板的影片與 `index.csv` | `--same_as <index.csv>` 會把另一個設定的同一批 episode 也畫一份，方便對照 |
| `annotate_chunks.py` | 單一影片標上 chunk / step | 原始影片為 80 fps，輸出預設 20 fps |
| `view_npz.py` | 印出 npz 的陣列名稱、形狀與數值 | |

## 輸出資料格式

在 `$OUTPUTS_DIR`（`outputs/` 或 `outputs_act/`）底下：

| 檔案 | 內容 |
|---|---|
| `plan.json` | 每個 suite 抽到的 task_id |
| `results.csv` | 每個 episode 一列：suite、task_id、task_name、category、difficulty、episode、success、bddl_label；**順序即所有分析陣列的順序** |
| `<suite>/chunk_XXXX/` | lerobot-eval 的原始輸出與影片（`videos/<suite>_<task_id>/eval_episode_0.mp4`） |
| `<suite>/chunk_XXXX_features/<suite>_<task_id>_ep<k>.npz` | 特徵（C = chunk 數）：`step` (C,)、`lm_mean` / `lm_last` / `act_mean` (C,18,2048)、`act_prelogits` (C,1,2048)、`n_act_tokens` (C,)、`vision_mean` (C,3,27,1152)；`act_*` 只在 `outputs_act` 才有 |
| `<suite>/chunk_XXXX_bddl/<同名>.npz` | `label`（True = 成功） |
| `chunk_features_<feature>.npy` | 分析用快取 (層, episode, Tmax, D) float16；特徵更新後要刪掉重建 |
| `episode_scores_*.csv / .npz` | `export_episode_scores.py` 的輸出 |
| `annotated_videos/` | 標註影片 |

## 執行環境注意事項

- **每個新 terminal 都要 `source activate.sh`**。分析程式用 `export OUTPUTS_DIR=outputs_act` 指定資料夾，預設為 `outputs`；路徑相對於目前所在目錄。
- **EGL**：EGL 函式庫必須與目前載入的驅動版本相同（目前為 595.91.07，使用系統的 libnvidia-gl）。驅動升級並重開機後，開新的 terminal 執行 `bash setup.sh --skip-assets --skip-model`。舊 terminal 殘留的 `__EGL_VENDOR_LIBRARY_FILENAMES` 會讓 EGL 找不到 GPU。
- **LIBERO-plus 修正**（Sensor Noise 擾動需要，`setup.sh` 會自動套用）：`libero/libero/envs/env_wrapper.py` 的 `np.float_` 改為 `np.float64`；`fog()` 的雜訊圖大小改為不小於影像（原本固定 256，lerobot 影像為 360）。不能把 NumPy 降到 1.x，lerobot 與 opencv 需要 numpy>=2。
- **GPU**：單張 RTX 3080 Ti（12GB）。評測時盡量不要同時跑其他 GPU 工作。
- **長時間工作**：完整評測要好幾個小時，請在 tmux 中執行。

## 已知結論與陷阱（修改前先讀）

- **初始資訊**：prefill 特徵只看第一個 chunk 就有 AUROC 0.83。比較方法時要看 `--input first`、逐時間 AUROC 在 t=1、2 是否接近 0.5，以及 `--split task`。
- **長度**：任何結果都要和報表中的「長度基準」比較。
- **cumsum + functional CP**：若有超過 α 比例的成功 episode 被判為 p≈1，門檻會貼近上限 t，結果會退化成長度基準。在 lm_mean 上 prefix 分數明顯較好（BA 0.75 對 0.66）。
- **浮點數平手**：警報判斷使用 `is_alarm()`（帶極小容差）；`export_episode_scores.py` 與偵測程式共用它，不要改回直接比較 `s > tau`。
- **舊資料**：`outputs/` 沒有解碼特徵，也缺 Sensor Noise。新的比較一律使用 `outputs_act/`。
- **git**：`.gitignore` 目前只排除 `eval_pi0fast/outputs/` 的大型檔案；`outputs_act/` 的特徵與影片很大，提交前請確認不會被加入。
