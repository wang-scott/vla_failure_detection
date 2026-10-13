#!/usr/bin/env bash
# 一鍵建立環境：micromamba + conda env + lerobot + LIBERO-plus + assets + 模型權重
# 用法: bash setup.sh [--nvidia-gl] [--skip-assets] [--skip-model]
set -euo pipefail

VLA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$VLA_ROOT"

# 與原始實驗相同的版本
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

step "1/7 下載 micromamba"
if [ ! -x bin/micromamba ]; then
    mkdir -p bin
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C . bin/micromamba
fi

step "2/7 建立 conda 環境 (env/)"
export MAMBA_ROOT_PREFIX="$VLA_ROOT/micromamba"
if [ ! -d env ]; then
    bin/micromamba create -y -p "$VLA_ROOT/env" -c conda-forge \
        python=3.12 expat fontconfig imagemagick pkg-config cmake make
fi

step "3/7 下載 lerobot 與 LIBERO-plus 原始碼"
clone_at() {  # clone_at <url> <dir> <commit>
    if [ ! -d "$2" ]; then git clone "$1" "$2"; fi
    git -C "$2" fetch -q origin "$3" 2>/dev/null || true
    git -C "$2" checkout -q "$3"
}
clone_at https://github.com/huggingface/lerobot.git lerobot "$LEROBOT_COMMIT"
clone_at https://github.com/sylvestf/LIBERO-plus.git LIBERO-plus "$LIBERO_PLUS_COMMIT"

step "4/7 安裝 Python 套件"
PIP="$VLA_ROOT/env/bin/python -m pip"
$PIP install -r requirements-lock.txt
$PIP install --no-deps -e ./lerobot -e ./LIBERO-plus
# 移除 hf-libero，避免 import libero 時抓到原版 LIBERO
$PIP uninstall -y hf-libero 2>/dev/null || true

step "5/7 寫入 LIBERO 設定 (~/.libero/config.yaml)"
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

step "6/7 下載 LIBERO-plus assets (~6GB，解壓後 ~9.5GB)"
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

step "7/7 下載模型權重 lerobot/pi0fast-libero 與 paligemma tokenizer"
if [ "$SKIP_MODEL" = 1 ]; then
    echo "略過"
else
    # google/paligemma-3b-pt-224 是 gated model，需先 `hf auth login` 並在網頁上同意授權
    env/bin/hf download lerobot/pi0fast-libero
    env/bin/python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('google/paligemma-3b-pt-224')" \
        || echo "!! paligemma tokenizer 下載失敗：請執行 'source activate.sh && hf auth login' 後重跑 'bash setup.sh --skip-assets'"
fi

if [ "$WITH_NVIDIA_GL" = 1 ]; then
    step "額外：下載與驅動同版本的 libnvidia-gl (無 EGL 的伺服器用)"
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    MAJOR=${DRIVER%%.*}
    rm -rf nvidia-gl && mkdir -p nvidia-gl && cd nvidia-gl
    apt download "libnvidia-gl-${MAJOR}-server" || apt download "libnvidia-gl-${MAJOR}"
    DEB=$(ls libnvidia-gl-*.deb | head -1)
    case "$DEB" in
        *"_${DRIVER}-"*) ;;
        *) echo "!! 下載到的 $DEB 與驅動版本 $DRIVER 不符，EGL 可能無法使用" >&2 ;;
    esac
    dpkg -x "$DEB" root && rm "$DEB"
    cd "$VLA_ROOT"
fi

echo -e "\n完成！之後每次使用前執行：  source $VLA_ROOT/activate.sh"
