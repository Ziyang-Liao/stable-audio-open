"""
Stable Audio Open - Trainium/Neuron 适配版本
用于与 GPU 版本进行性能对比测试
"""
import torch
import torchaudio
import argparse
import time
import os
from einops import rearrange

# Neuron SDK
import torch_neuronx

def get_memory_info():
    """获取内存使用情况"""
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**3

class NeuronDiffusionWrapper(torch.nn.Module):
    """包装 diffusion model 用于 Neuron 编译"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x, t, cond):
        return self.model(x, t, cond=cond)

def compile_model_for_neuron(model, model_config):
    """编译模型到 Neuron"""
    print("编译模型到 Neuron...")
    
    # 获取 diffusion backbone
    diffusion = model.model.model.diffusion
    
    # 创建示例输入
    sample_size = model_config["sample_size"]
    # stable-audio-open 使用 64 channels latent
    example_x = torch.randn(1, 64, sample_size // 2048)
    example_t = torch.tensor([1.0])
    example_cond = {"cross_attn_cond": torch.randn(1, 512, 768)}
    
    # 编译
    wrapper = NeuronDiffusionWrapper(diffusion)
    try:
        compiled = torch_neuronx.trace(
            wrapper,
            (example_x, example_t, example_cond),
            compiler_args=["--auto-cast", "all", "--auto-cast-type", "bf16"]
        )
        print("Neuron 编译成功!")
        return compiled
    except Exception as e:
        print(f"Neuron 编译失败: {e}")
        print("使用 CPU 回退模式")
        return None

def generate_audio_neuron(model, model_config, prompt, duration, steps=100, cfg_scale=7, compiled_diffusion=None):
    """使用 Neuron 生成音频"""
    from stable_audio_tools.inference.generation import generate_diffusion_cond
    
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    max_duration = sample_size / sample_rate
    actual_duration = min(duration, max_duration)
    
    conditioning = [{
        "prompt": prompt,
        "seconds_start": 0,
        "seconds_total": actual_duration
    }]
    
    mem_before = get_memory_info()
    start_time = time.time()
    
    # 使用 CPU 进行推理 (Neuron 设备会自动处理)
    device = "xla" if compiled_diffusion else "cpu"
    
    output = generate_diffusion_cond(
        model,
        steps=steps,
        cfg_scale=cfg_scale,
        conditioning=conditioning,
        sample_size=sample_size,
        sigma_min=0.3,
        sigma_max=500,
        sampler_type="dpmpp-3m-sde",
        device=device
    )
    
    gen_time = time.time() - start_time
    mem_after = get_memory_info()
    mem_used = mem_after - mem_before
    
    # 后处理
    output = rearrange(output, "b d n -> d (b n)")
    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1)
    output = (output * 32767).to(torch.int16).cpu()
    
    return output, sample_rate, gen_time, mem_used, actual_duration

def run_benchmark(model, model_config, iterations=5, duration=30, steps=100, compiled_diffusion=None):
    """运行性能测试"""
    prompts = [
        "128 BPM tech house drum loop",
        "ambient soundscape with rain", 
        "cinematic orchestral music",
        "electronic bass drop",
        "acoustic guitar melody"
    ]
    
    results = []
    device_name = "Trainium" if compiled_diffusion else "CPU"
    
    print(f"\n{'='*60}")
    print(f"[{device_name}] 性能测试: {iterations} 次迭代, {duration}s 音频, {steps} 步")
    print(f"{'='*60}\n")
    
    # 预热
    print("预热中...")
    _ = generate_audio_neuron(model, model_config, "warmup", 5, steps=10, compiled_diffusion=compiled_diffusion)
    
    for i in range(iterations):
        prompt = prompts[i % len(prompts)]
        print(f"[{i+1}/{iterations}] 生成: {prompt[:40]}...")
        
        output, sr, gen_time, mem_used, actual_dur = generate_audio_neuron(
            model, model_config, prompt, duration, steps, compiled_diffusion=compiled_diffusion
        )
        
        rtf = gen_time / actual_dur
        results.append({
            'time': gen_time,
            'memory': mem_used,
            'rtf': rtf,
            'duration': actual_dur
        })
        
        print(f"    时间: {gen_time:.2f}s | 内存: {mem_used:.2f}GB | RTF: {rtf:.2f}x")
    
    # 统计
    avg_time = sum(r['time'] for r in results) / len(results)
    avg_mem = sum(r['memory'] for r in results) / len(results)
    avg_rtf = sum(r['rtf'] for r in results) / len(results)
    
    print(f"\n{'='*60}")
    print(f"[{device_name}] 测试结果汇总")
    print(f"{'='*60}")
    print(f"平均生成时间: {avg_time:.2f}s")
    print(f"平均内存增量: {avg_mem:.2f}GB")
    print(f"平均实时因子: {avg_rtf:.2f}x (越小越快)")
    print(f"{'='*60}\n")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Stable Audio Open - Neuron 性能测试')
    parser.add_argument('--prompt', type=str, default='ambient soundscape')
    parser.add_argument('--duration', type=int, default=30)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--cfg_scale', type=float, default=7)
    parser.add_argument('--output', type=str, default='output_neuron.wav')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--no-compile', action='store_true', help='跳过 Neuron 编译,使用 CPU')
    args = parser.parse_args()
    
    # 检查 Neuron 设备
    print("检查 Neuron 设备...")
    try:
        import torch_neuronx
        neuron_available = True
        print(f"Neuron SDK 版本: {torch_neuronx.__version__}")
    except ImportError:
        neuron_available = False
        print("Neuron SDK 不可用, 使用 CPU 模式")
    
    # 加载模型
    print("\n加载模型...")
    load_start = time.time()
    
    from stable_audio_tools import get_pretrained_model
    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    
    load_time = time.time() - load_start
    print(f"模型加载完成: {load_time:.2f}s")
    print(f"采样率: {model_config['sample_rate']}Hz")
    print(f"最大时长: {model_config['sample_size'] / model_config['sample_rate']:.1f}s")
    
    # 编译到 Neuron
    compiled_diffusion = None
    if neuron_available and not args.no_compile:
        compiled_diffusion = compile_model_for_neuron(model, model_config)
    
    if args.benchmark:
        run_benchmark(model, model_config, args.iterations, args.duration, args.steps, compiled_diffusion)
    else:
        print(f"\n生成音频: {args.prompt}")
        output, sr, gen_time, mem_used, actual_dur = generate_audio_neuron(
            model, model_config, args.prompt, args.duration, args.steps, args.cfg_scale, compiled_diffusion
        )
        
        torchaudio.save(args.output, output, sr)
        print(f"\n生成完成!")
        print(f"  文件: {args.output}")
        print(f"  时长: {actual_dur:.1f}s")
        print(f"  生成时间: {gen_time:.2f}s")
        print(f"  实时因子: {gen_time/actual_dur:.2f}x")

if __name__ == "__main__":
    main()
