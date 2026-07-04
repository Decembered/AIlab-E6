#!/bin/bash
set -e
echo "Installing PyTorch from Tsinghua mirror..."
pip install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple
echo "Done."
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
