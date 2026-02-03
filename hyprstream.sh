#!/bin/bash
# hyprstream.sh - Launch script with sensible defaults and ENV overrides
#
# Supports both NVIDIA CUDA and AMD ROCm backends. Auto-detects based on
# available libraries, or set HYPRSTREAM_GPU_BACKEND=cuda|rocm|cpu to override.
#
# For pip-installed PyTorch, set LIBTORCH to the torch package directory:
#   export LIBTORCH=$VIRTUAL_ENV/lib/python3.x/site-packages/torch

#export RUST_LOG=warn

# Server configuration (all overridable via ENV)
export HYPRSTREAM_SERVER_HOST=${HYPRSTREAM_SERVER_HOST:-0.0.0.0}
export HYPRSTREAM_CORS_ENABLED=${HYPRSTREAM_CORS_ENABLED:-true}
export HYPRSTREAM_CORS_ORIGINS=${HYPRSTREAM_CORS_ORIGINS:-"*"}

# Paths - relative to script location by default, override with ENV
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPS_DIR=${DEPS_DIR:-$SCRIPT_DIR/target-deps}

# Libtorch location - can be standalone libtorch or pip-installed torch package
export LIBTORCH=${LIBTORCH:-$DEPS_DIR/libtorch}

# Determine library directory (handles both standalone and pip-installed torch)
if [ -d "$LIBTORCH/lib" ]; then
    TORCH_LIB="$LIBTORCH/lib"
else
    TORCH_LIB="$LIBTORCH"
fi

# Bitsandbytes library directory (for LD_LIBRARY_PATH)
if [ -n "$BITSANDBYTES_LIB_PATH" ] && [ -f "$BITSANDBYTES_LIB_PATH" ]; then
    BITSANDBYTES_LIB_DIR=$(dirname "$BITSANDBYTES_LIB_PATH")
else
    BITSANDBYTES_LIB_DIR=${BITSANDBYTES_LIB_PATH:-$DEPS_DIR/bitsandbytes/bitsandbytes}
fi

# Libtorch build settings
export LIBTORCH_STATIC=0
export LIBTORCH_BYPASS_VERSION_CHECK=1

# Auto-detect GPU backend if not specified
if [ -z "$HYPRSTREAM_GPU_BACKEND" ]; then
    if [ -f "$TORCH_LIB/libtorch_cuda.so" ]; then
        HYPRSTREAM_GPU_BACKEND="cuda"
    elif [ -f "$TORCH_LIB/libtorch_hip.so" ]; then
        HYPRSTREAM_GPU_BACKEND="rocm"
    else
        HYPRSTREAM_GPU_BACKEND="cpu"
    fi
fi

# Base library path
export LD_LIBRARY_PATH="$TORCH_LIB:$BITSANDBYTES_LIB_DIR:$LD_LIBRARY_PATH"

# Backend-specific configuration
case "$HYPRSTREAM_GPU_BACKEND" in
    cuda)
        # NVIDIA CUDA backend
        # LD_PRELOAD is required for CUDA detection with pip-installed PyTorch
        # (libtorch_cuda.so is lazily loaded and torch::cuda::is_available()
        # returns false unless we force-load it)
        if [ -f "$TORCH_LIB/libtorch_cuda.so" ]; then
            export LD_PRELOAD="$TORCH_LIB/libtorch_cuda.so"
        fi

        # For pip-installed PyTorch, add nvidia package library paths
        # These are installed as separate pip packages (nvidia-cublas-cu12, etc.)
        NVIDIA_LIB="$LIBTORCH/../nvidia"
        if [ -d "$NVIDIA_LIB" ]; then
            for subdir in cublas cuda_cupti cuda_nvrtc cuda_runtime cudnn cufft curand cusolver cusparse nccl nvjitlink; do
                if [ -d "$NVIDIA_LIB/$subdir/lib" ]; then
                    export LD_LIBRARY_PATH="$NVIDIA_LIB/$subdir/lib:$LD_LIBRARY_PATH"
                fi
            done
        fi
        ;;

    rocm)
        # AMD ROCm backend
        export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
        export PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH:-gfx90a}
        ;;

    cpu)
        # CPU-only, no additional configuration needed
        ;;

    *)
        echo "Warning: Unknown GPU backend '$HYPRSTREAM_GPU_BACKEND', using cpu" >&2
        ;;
esac

# Run hyprstream with all passed arguments
exec "$SCRIPT_DIR/target/release/hyprstream" "$@"
