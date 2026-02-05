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

| GPU | 30s 音频生成时间 | 显存占用 |
|-----|-----------------|---------|
| NVIDIA L40S (46GB) | TBD | TBD |

## License

本项目仅用于性能测试和研究目的。模型使用需遵循 [Stability AI Community License](https://huggingface.co/stabilityai/stable-audio-open-1.0/blob/main/LICENSE.md)。
