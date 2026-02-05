#!/usr/bin/env python3
"""
Stable Audio Open - GPU 优化版本
使用 torch.compile 和 CUDA 优化，不降低精度
"""
import torch
import torchaudio
import time
import argparse
from einops import rearrange

def optimize_model(model):
    """优化模型（不改变精度）"""
    # 1. torch.compile - 融合算子，减少 kernel launch
    if hasattr(torch, 'compile'):
        print("应用 torch.compile 优化...")
        model.model.model = torch.compile(
            model.model.model,
            mode="reduce-overhead",  # 减少开销，不改变数值
            fullgraph=False
        )
    return model

def generate_optimized(model, model_config, prompt, duration=30, steps=100, cfg_scale=7):
    """优化的生成函数"""
    from stable_audio_tools.inference.generation import generate_diffusion_cond
    
    device = "cuda"
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    
    conditioning = [{
        "prompt": prompt,
        "seconds_start": 0,
        "seconds_total": min(duration, sample_size / sample_rate)
    }]
    
    # 预热（让 compile 生效）
    if hasattr(model.model.model, '_compiled'):
        print("预热编译模型...")
        with torch.no_grad():
            _ = generate_diffusion_cond(
                model, steps=2, cfg_scale=cfg_scale,
                conditioning=conditioning, sample_size=sample_size,
                sigma_min=0.3, sigma_max=500,
                sampler_type="dpmpp-3m-sde", device=device
            )
        torch.cuda.synchronize()
    
    # 正式生成
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.time()
    
    with torch.no_grad():
        output = generate_diffusion_cond(
            model, steps=steps, cfg_scale=cfg_scale,
            conditioning=conditioning, sample_size=sample_size,
            sigma_min=0.3, sigma_max=500,
            sampler_type="dpmpp-3m-sde", device=device
        )
    
    torch.cuda.synchronize()
    gen_time = time.time() - start
    mem = torch.cuda.max_memory_allocated() / 1024**3
    
    # 后处理
    output = rearrange(output, "b d n -> d (b n)")
    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1)
    output = (output * 32767).to(torch.int16).cpu()
    
    return output, sample_rate, gen_time, mem

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="128 BPM tech house drum loop")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", default="output_optimized.wav")
    parser.add_argument("--no-compile", action="store_true", help="禁用 torch.compile")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # 加载模型
    print("\n加载模型...")
    from stable_audio_tools import get_pretrained_model
    model, config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    model = model.to(device)
    
    # 优化
    if not args.no_compile:
        model = optimize_model(model)
    
    if args.benchmark:
        print(f"\n{'='*50}")
        print(f"Benchmark: {args.iterations} 次, {args.steps} 步")
        print(f"{'='*50}\n")
        
        times = []
        for i in range(args.iterations):
            print(f"[{i+1}/{args.iterations}]...")
            _, _, t, mem = generate_optimized(model, config, args.prompt, steps=args.steps)
            times.append(t)
            print(f"  时间: {t:.2f}s, 显存: {mem:.2f}GB")
        
        avg = sum(times) / len(times)
        # 跳过第一次（预热）
        avg_warm = sum(times[1:]) / len(times[1:]) if len(times) > 1 else avg
        print(f"\n平均: {avg:.2f}s (预热后: {avg_warm:.2f}s)")
        print(f"RTF: {avg_warm/30:.2f}x")
    else:
        print(f"\n生成: {args.prompt}")
        output, sr, t, mem = generate_optimized(model, config, args.prompt, steps=args.steps)
        torchaudio.save(args.output, output, sr)
        print(f"\n完成! 时间: {t:.2f}s, 显存: {mem:.2f}GB")

if __name__ == "__main__":
    main()
