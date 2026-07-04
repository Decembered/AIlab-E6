#!/bin/bash
set -e
echo "Removing all CUDA packages..."
pip uninstall torch torchvision triton -y 2>/dev/null
for pkg in nvidia-cublas nvidia-cudnn nvidia-cusparselt nvidia-nccl nvidia-nvshmem nvidia-cuda-runtime nvidia-cufft nvidia-cufile nvidia-cuda-cupti nvidia-curand nvidia-cusolver nvidia-cusparse nvidia-nvjitlink nvidia-cuda-nvrtc nvidia-nvtx cuda-bindings cuda-toolkit cuda-pathfinder; do
    pip uninstall $pkg -y 2>/dev/null
done
echo "Installing CPU-only PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
echo "Testing..."
python -c "import torch; print(f'PyTorch {torch.__version__}'); print('OK! CPU mode.')"
