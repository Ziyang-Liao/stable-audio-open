#!/bin/bash
# Trainium 实例初始化脚本
# 在 trn1 实例上运行此脚本设置环境

set -e

echo "=== Trainium 环境设置 ==="

# 激活 Neuron 虚拟环境
source /opt/aws_neuronx_venv_pytorch_2_5_nxd_inference/bin/activate

# 安装依赖
echo "安装依赖..."
pip install -q einops torchaudio psutil

# 安装 stable-audio-tools
echo "安装 stable-audio-tools..."
pip install -q stable-audio-tools

# 验证 Neuron
echo ""
echo "=== 环境验证 ==="
python -c "import torch_neuronx; print(f'Neuron SDK: {torch_neuronx.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
neuron-ls

echo ""
echo "=== 设置完成 ==="
echo "运行测试: python benchmark_neuron.py --benchmark --iterations 3"
