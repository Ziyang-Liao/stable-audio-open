#!/bin/bash
# 性能对比测试脚本
# 分别在 GPU 和 Trainium 上运行后对比结果

ITERATIONS=5
STEPS=100
DURATION=30

echo "=============================================="
echo "Stable Audio Open 性能对比测试"
echo "=============================================="
echo "迭代次数: $ITERATIONS"
echo "推理步数: $STEPS"  
echo "音频时长: ${DURATION}s"
echo ""

# 检测设备类型
if command -v neuron-ls &> /dev/null; then
    echo "检测到 Trainium 设备"
    source /opt/aws_neuronx_venv_pytorch_2_5_nxd_inference/bin/activate 2>/dev/null || true
    python benchmark_neuron.py --benchmark --iterations $ITERATIONS --steps $STEPS --duration $DURATION
elif python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo "检测到 CUDA GPU"
    python benchmark.py --benchmark --iterations $ITERATIONS --steps $STEPS --duration $DURATION
else
    echo "未检测到加速设备,使用 CPU"
    python benchmark_neuron.py --benchmark --iterations $ITERATIONS --steps $STEPS --duration $DURATION --no-compile
fi
