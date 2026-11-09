# 使用方式: source <repo>/activate.sh
VLA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VLA_ROOT
export MAMBA_ROOT_PREFIX="$VLA_ROOT/micromamba"
eval "$("$VLA_ROOT/bin/micromamba" shell hook -s bash)"
micromamba activate "$VLA_ROOT/env"
export MAGICK_HOME="$CONDA_PREFIX"
# torchvision 內建的 libjpeg.so.8 缺少 jpeg12_* 符號，會讓 ImageMagick(Wand) 載入失敗；
# 先載入 conda 的 libjpeg-turbo 讓所有函式庫共用它
export LD_PRELOAD="$CONDA_PREFIX/lib/libjpeg.so.8${LD_PRELOAD:+:$LD_PRELOAD}"
# LIBERO-plus 以 namespace package 安裝，需要放進 PYTHONPATH
export PYTHONPATH="$VLA_ROOT/LIBERO-plus${PYTHONPATH:+:$PYTHONPATH}"
# 無螢幕伺服器使用 EGL 算圖
export MUJOCO_GL=egl
# 伺服器只裝了 NVIDIA 計算用驅動、沒有 EGL 時，改用解壓到 nvidia-gl/ 的
# libnvidia-gl-<版本>-server（版本必須與驅動相同，見 setup.sh --nvidia-gl），驅動升級後需重新下載
if [ -d "$VLA_ROOT/nvidia-gl/root" ]; then
    export __EGL_VENDOR_LIBRARY_FILENAMES="$VLA_ROOT/nvidia-gl/root/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    export LD_LIBRARY_PATH="$VLA_ROOT/nvidia-gl/root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
elif [[ "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" == "$VLA_ROOT/nvidia-gl/"* ]]; then
    # nvidia-gl/ 已移除（例如驅動升級後改用系統的 EGL），清掉之前 source 留下的設定，否則 EGL 找不到任何 GPU
    unset __EGL_VENDOR_LIBRARY_FILENAMES
    LD_LIBRARY_PATH="$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v "$VLA_ROOT/nvidia-gl/" | paste -sd:)"
    export LD_LIBRARY_PATH
fi
