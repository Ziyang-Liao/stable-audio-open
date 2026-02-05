# Stable Audio Open 性能测试

基于 [stabilityai/stable-audio-open-1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0) 的性能测试项目。

## 环境要求

- NVIDIA GPU (推荐 L40S 或更高)
- CUDA 12.0+
- Python 3.10+

## 安装

```bash
# 安装 stable-audio-tools
pip install stable-audio-tools

# 安装其他依赖
pip install einops torchaudio
```

## 使用方法

### 基础生成测试

```bash
python benchmark.py --prompt "128 BPM tech house drum loop" --duration 30
```

### 性能测试

```bash
python benchmark.py --benchmark --iterations 10
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --prompt | "ambient soundscape" | 生成提示词 |
| --duration | 30 | 音频时长(秒) |
| --steps | 100 | 推理步数 |
| --cfg_scale | 7 | CFG 引导强度 |
| --benchmark | False | 运行性能测试 |
| --iterations | 5 | 测试迭代次数 |

## 性能指标

测试输出包括：
- 首次生成时间 (含模型加载)
- 平均生成时间
- GPU 显存占用
- 实时因子 (RTF)

## 硬件测试结果

| 实例类型 | GPU | 显存 | vCPU | 内存 | 30s 音频生成 | 显存占用 | RTF | 相对性能 |
|---------|-----|------|------|------|-------------|---------|-----|---------|
| - | NVIDIA L40S | 46GB | - | - | 17.62s | 9.61GB | 0.59x | 基准 (1.0x) |
| g5.2xlarge | NVIDIA A10G | 23GB | 8 | 32GB | 34.41s | 9.61GB | 1.15x | 0.51x |

> RTF (Real-Time Factor): 生成时间/音频时长，越小越快。0.59x 表示生成 30s 音频只需 17.6s。

**性能分析：**
- 模型参数：1.2B (12亿)
- 最小显存需求：~10GB
- 性能瓶颈：GPU 计算能力（Tensor Core FP16）
- A10G 性能约为 L40S 的 51%，符合硬件算力差距（L40S FP16: 90.5 TFLOPS vs A10G: 31.2 TFLOPS）

## 并发测试结论

| 并行进程数 | 总耗时 | 生成音频 | 吞吐量 |
|-----------|--------|---------|--------|
| 1 (顺序) | 17.6s | 30s | **1.70x 实时** |
| 2 | 53.7s | 60s | 1.12x 实时 |
| 3 | 74.1s | 90s | 1.21x 实时 |

**结论：不适合跑并发**

- 多进程并行时 GPU 资源互相竞争，每个进程都变慢
- 单进程效率最高 (1.70x)，并发反而下降 30%+
- 显存只用 9.6GB/46GB (21%)，但计算单元已饱和

**最优方案：单进程顺序执行 + 请求队列**

## License

本项目仅用于性能测试和研究目的。模型使用需遵循 [Stability AI Community License](https://huggingface.co/stabilityai/stable-audio-open-1.0/blob/main/LICENSE.md)。
