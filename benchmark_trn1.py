"""
Stable Audio Open - Trainium Neuron 推理
使用 torch_neuronx 编译 diffusion 模型核心组件
"""
import torch
import torch_neuronx
import torchaudio
import time
import os
import argparse
from einops import rearrange

os.environ["NEURON_RT_NUM_CORES"] = "2"

def trace_unet_for_neuron(model, sample_size):
    """编译 UNet/DiT 到 Neuron"""
    print("编译 diffusion backbone 到 Neuron...")
    
    diffusion = model.model.model.diffusion
    
    # 获取模型配置
    latent_dim = 64  # stable-audio-open latent channels
    seq_len = sample_size // 2048
    
    # 创建示例输入
    example_x = torch.randn(2, latent_dim, seq_len)  # batch=2 for CFG
    example_t = torch.ones(2)
    
    # 包装前向函数
    class DiffusionWrapper(torch.nn.Module):
        def __init__(self, diffusion):
            super().__init__()
            self.diffusion = diffusion
            
        def forward(self, x, t):
            return self.diffusion(x, t)
    
    wrapper = DiffusionWrapper(diffusion).eval()
    
    # 编译到 Neuron
    compiled = torch_neuronx.trace(
        wrapper,
        (example_x, example_t),
        compiler_args=["--auto-cast", "all", "--auto-cast-type", "bf16", "--model-type", "transformer"]
    )
    
    print("Neuron 编译完成!")
    return compiled

def generate_with_neuron(model, model_config, prompt, duration, steps, compiled_diffusion):
    """使用 Neuron 编译的模型生成音频"""
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
    
    # 替换原始 diffusion 为编译版本
    if compiled_diffusion is not None:
        original_diffusion = model.model.model.diffusion
        model.model.model.diffusion = compiled_diffusion
    
    start_time = time.time()
    
    output = generate_diffusion_cond(
        model,
        steps=steps,
        cfg_scale=7,
        conditioning=conditioning,
        sample_size=sample_size,
        sigma_min=0.3,
        sigma_max=500,
        sampler_type="dpmpp-3m-sde",
        device="cpu"  # Neuron 通过 XLA 自动处理
    )
    
    gen_time = time.time() - start_time
    
    # 恢复原始模型
    if compiled_diffusion is not None:
        model.model.model.diffusion = original_diffusion
    
    # 后处理
    output = rearrange(output, "b d n -> d (b n)")
    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1)
    output = (output * 32767).to(torch.int16).cpu()
    
    return output, sample_rate, gen_time, actual_duration

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--duration', type=int, default=30)
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--output', type=str, default='output_trn1.wav')
    args = parser.parse_args()
    
    print(f"Neuron SDK: {torch_neuronx.__version__}")
    
    # 加载模型
    print("\n加载模型...")
    from stable_audio_tools import get_pretrained_model
    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    print(f"采样率: {model_config['sample_rate']}Hz")
    
    # 编译到 Neuron
    try:
        compiled = trace_unet_for_neuron(model, model_config["sample_size"])
    except Exception as e:
        print(f"Neuron 编译失败: {e}")
        compiled = None
    
    # 运行测试
    prompts = ["128 BPM tech house drum loop", "ambient soundscape", "cinematic orchestra"]
    results = []
    
    print(f"\n{'='*50}")
    print(f"Trainium 性能测试: {args.iterations} 次, {args.steps} 步")
    print(f"{'='*50}\n")
    
    for i in range(args.iterations):
        prompt = prompts[i % len(prompts)]
        print(f"[{i+1}/{args.iterations}] {prompt[:30]}...")
        
        output, sr, gen_time, dur = generate_with_neuron(
            model, model_config, prompt, args.duration, args.steps, compiled
        )
        
        rtf = gen_time / dur
        results.append({'time': gen_time, 'rtf': rtf})
        print(f"    时间: {gen_time:.2f}s | RTF: {rtf:.2f}x")
        
        if i == 0:
            torchaudio.save(args.output, output, sr)
    
    avg_time = sum(r['time'] for r in results) / len(results)
    avg_rtf = sum(r['rtf'] for r in results) / len(results)
    
    print(f"\n{'='*50}")
    print(f"平均生成时间: {avg_time:.2f}s")
    print(f"平均 RTF: {avg_rtf:.2f}x")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
