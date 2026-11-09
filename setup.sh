#!/usr/bin/env bash
# 一鍵建立環境：micromamba + conda env + lerobot + LIBERO-plus + assets + 模型權重 + EGL 檢查
# 用法: bash setup.sh [--nvidia-gl] [--skip-assets] [--skip-model]
#   --nvidia-gl  強制下載與驅動同版本的 libnvidia-gl 到 nvidia-gl/（預設會自動判斷：系統已有同版本 EGL 就直接使用）
# 已建立的環境重跑此腳本是安全的：已完成的步驟會略過，Python 套件會同步到 requirements-lock.txt
set -euo pipefail

VLA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$VLA_ROOT"

# 與最新實驗相同的版本（requirements-lock.txt 亦為最新實驗環境的 pip freeze）
LEROBOT_COMMIT=ff71cae1ae2d09fd035553c35da65888ed6c8304
LIBERO_PLUS_COMMIT=4976dc30028e805ff8094b55501d532c48fec182

WITH_NVIDIA_GL=0
SKIP_ASSETS=0
SKIP_MODEL=0
for arg in "$@"; do
    case "$arg" in
        --nvidia-gl) WITH_NVIDIA_GL=1 ;;
        --skip-assets) SKIP_ASSETS=1 ;;
        --skip-model) SKIP_MODEL=1 ;;
        *) echo "未知參數: $arg" >&2; exit 1 ;;
    esac
done

step() { echo -e "\n==> $*"; }

step "1/8 下載 micromamba"
if [ ! -x bin/micromamba ]; then
    mkdir -p bin
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C . bin/micromamba
fi

step "2/8 建立 conda 環境 (env/)"
export MAMBA_ROOT_PREFIX="$VLA_ROOT/micromamba"
if [ ! -d env ]; then
    bin/micromamba create -y -p "$VLA_ROOT/env" -c conda-forge \
        python=3.12 expat fontconfig imagemagick pkg-config cmake make
fi

step "3/8 下載 lerobot 與 LIBERO-plus 原始碼"
clone_at() {  # clone_at <url> <dir> <commit>
    if [ ! -d "$2" ]; then git clone "$1" "$2"; fi
    git -C "$2" fetch -q origin "$3" 2>/dev/null || true
    git -C "$2" checkout -q "$3"
}
clone_at https://github.com/huggingface/lerobot.git lerobot "$LEROBOT_COMMIT"
clone_at https://github.com/sylvestf/LIBERO-plus.git LIBERO-plus "$LIBERO_PLUS_COMMIT"
# LIBERO-plus 的 Sensor Noise 擾動（fog → plasma_fractal）使用 NumPy 2 已移除的 np.float_，
# 而 lerobot 需要 numpy>=2，因此改成等價的 np.float64（np.float_ 原本就是它的別名）；重複執行不影響
sed -i 's/dtype=np\.float_)/dtype=np.float64)/' LIBERO-plus/libero/libero/envs/env_wrapper.py
# 同一擾動的 fog 固定產生 256x256 的雜訊圖，lerobot 的影像為 360x360 時會 broadcast 失敗；
# 改成不小於影像的 2 的次方（至少 256，影像 <= 256 時結果不變）；重複執行不影響
sed -i 's/    x += c\[0\] \* plasma_fractal(wibbledecay=c\[1\])/    mapsize = max(256, 1 << (max(height_x, weight_x) - 1).bit_length())\n    x += c[0] * plasma_fractal(mapsize=mapsize, wibbledecay=c[1])/' \
    LIBERO-plus/libero/libero/envs/env_wrapper.py

step "4/8 安裝 Python 套件"
PIP="$VLA_ROOT/env/bin/python -m pip"
$PIP install -r requirements-lock.txt
$PIP install --no-deps -e ./lerobot -e ./LIBERO-plus
# 移除 hf-libero，避免 import libero 時抓到原版 LIBERO
$PIP uninstall -y hf-libero 2>/dev/null || true

step "5/8 寫入 LIBERO 設定 (~/.libero/config.yaml)"
LIBERO_DIR="$VLA_ROOT/LIBERO-plus/libero/libero"
mkdir -p ~/.libero
if [ -f ~/.libero/config.yaml ]; then
    cp ~/.libero/config.yaml ~/.libero/config.yaml.bak
    echo "已備份舊設定到 ~/.libero/config.yaml.bak"
fi
cat > ~/.libero/config.yaml <<EOF
assets: $LIBERO_DIR/assets
bddl_files: $LIBERO_DIR/bddl_files
datasets: $LIBERO_DIR/../datasets
init_states: $LIBERO_DIR/init_files
benchmark_root: $LIBERO_DIR
EOF

step "6/8 下載 LIBERO-plus assets (~6GB，解壓後 ~9.5GB)"
if [ "$SKIP_ASSETS" = 1 ]; then
    echo "略過"
elif [ -d "$LIBERO_DIR/assets" ] && [ -n "$(ls -A "$LIBERO_DIR/assets")" ]; then
    echo "assets 已存在，略過"
else
    mkdir -p downloads
    wget -c -O downloads/assets.zip \
        https://huggingface.co/datasets/Sylvest/LIBERO-plus/resolve/main/assets.zip
    unzip -q downloads/assets.zip -d "$LIBERO_DIR"
    rm downloads/assets.zip
fi

step "7/8 下載模型權重 lerobot/pi0fast-libero 與 paligemma tokenizer"
if [ "$SKIP_MODEL" = 1 ]; then
    echo "略過"
else
    # google/paligemma-3b-pt-224 是 gated model，需先 `hf auth login` 並在網頁上同意授權
    env/bin/hf download lerobot/pi0fast-libero
    env/bin/python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('google/paligemma-3b-pt-224')" \
        || echo "!! paligemma tokenizer 下載失敗：請執行 'source activate.sh && hf auth login' 後重跑 'bash setup.sh --skip-assets'"
fi

step "8/8 EGL（無螢幕算圖）"
# 驅動升級並重開機後，nvidia-gl/ 內舊版本的 libnvidia-gl 會與新驅動不符，EGL 便找不到 GPU（模擬器無法啟動）。
# 系統已安裝同版本的 libEGL_nvidia 時直接使用系統的，並把過期的 nvidia-gl/ 改名保留。
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
nvgl_version() {  # nvidia-gl/ 內的 libEGL_nvidia 版本（不存在則為空）
    ls nvidia-gl/root/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.*.* 2>/dev/null | head -1 | sed 's/.*libEGL_nvidia\.so\.//'
}
if [ "$WITH_NVIDIA_GL" = 0 ] && ls /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so."$DRIVER" >/dev/null 2>&1; then
    echo "系統已有與驅動同版本（$DRIVER）的 EGL，直接使用"
    if [ -d nvidia-gl ]; then
        OLD="nvidia-gl.old-$(nvgl_version || echo unknown)"
        mv nvidia-gl "$OLD"
        echo "已將不再需要的 nvidia-gl/ 改名為 $OLD"
    fi
elif [ -d nvidia-gl ] && [ "$(nvgl_version)" = "$DRIVER" ] && [ "$WITH_NVIDIA_GL" = 0 ]; then
    echo "nvidia-gl/ 與驅動版本（$DRIVER）相同，沿用"
else
    echo "下載與驅動同版本（$DRIVER）的 libnvidia-gl 到 nvidia-gl/"
    MAJOR=${DRIVER%%.*}
    [ -d nvidia-gl ] && mv nvidia-gl "nvidia-gl.old-$(nvgl_version || echo unknown)"
    mkdir -p nvidia-gl && cd nvidia-gl
    apt download "libnvidia-gl-${MAJOR}-server" || apt download "libnvidia-gl-${MAJOR}"
    DEB=$(ls libnvidia-gl-*.deb | head -1)
    case "$DEB" in
        *"_${DRIVER}-"*) ;;
        *) echo "!! 下載到的 $DEB 與驅動版本 $DRIVER 不符，EGL 可能無法使用" >&2 ;;
    esac
    dpkg -x "$DEB" root && rm "$DEB"
    cd "$VLA_ROOT"
fi

# 用全新的子 shell 啟用環境並檢查（避免沿用目前 shell 中舊的 EGL 環境變數）
env -u __EGL_VENDOR_LIBRARY_FILENAMES bash -c '
    source activate.sh >/dev/null
    python - <<PY
import lerobot, libero.libero, cv2, imageio, openpyxl, wand.image  # noqa: F401
from mujoco.egl import egl_ext
n = len(egl_ext.eglQueryDevicesEXT())
from robosuite.renderers.context.egl_context import EGLGLContext
EGLGLContext(max_width=64, max_height=64, device_id=0)
print(f"套件 import 正常；EGL 看到 {n} 個裝置，robosuite 算圖正常")
PY
' || echo "!! 環境檢查失敗：請看上方錯誤；若是 EGL 問題，開新的 terminal 後 source activate.sh 再試，或執行 bash setup.sh --nvidia-gl --skip-assets --skip-model" >&2

echo -e "\n完成！之後每次使用前（請開新的 terminal）執行：  source $VLA_ROOT/activate.sh"
echo "重跑評估與失敗偵測的指令見 $VLA_ROOT/rerun_commands.txt"
