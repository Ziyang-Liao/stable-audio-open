"""
Stable Audio Open 性能测试脚本
使用 stable-audio-tools 库
"""
import torch
import torchaudio
import argparse
import time
import os
from einops import rearrange

def get_gpu_memory():
    """获取GPU显存使用情况"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**3
    return 0

def generate_audio(model, model_config, prompt, duration, steps=100, cfg_scale=7):
    """生成音频"""
    from stable_audio_tools.inference.generation import generate_diffusion_cond
    
    device = next(model.parameters()).device
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    
    # 计算实际采样点数
    max_duration = sample_size / sample_rate
    actual_duration = min(duration, max_duration)
    
    conditioning = [{
        "prompt": prompt,
        "seconds_start": 0,
        "seconds_total": actual_duration
    }]
    
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    
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
    gpu_mem = get_gpu_memory()
    
    # 后处理
    output = rearrange(output, "b d n -> d (b n)")
    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1)
    output = (output * 32767).to(torch.int16).cpu()
    
    return output, sample_rate, gen_time, gpu_mem, actual_duration

def run_benchmark(model, model_config, iterations=5, duration=30, steps=100):
    """运行性能测试"""
    prompts = [
        "128 BPM tech house drum loop",
        "ambient soundscape with rain",
        "cinematic orchestral music",
        "electronic bass drop",
        "acoustic guitar melody"
    ]
    
    results = []
    print(f"\n{'='*60}")
    print(f"性能测试: {iterations} 次迭代, {duration}s 音频, {steps} 步")
    print(f"{'='*60}\n")
    
    for i in range(iterations):
        prompt = prompts[i % len(prompts)]
        print(f"[{i+1}/{iterations}] 生成: {prompt[:40]}...")
        
        output, sr, gen_time, gpu_mem, actual_dur = generate_audio(
            model, model_config, prompt, duration, steps
        )
        
        rtf = gen_time / actual_dur  # 实时因子
        results.append({
            'time': gen_time,
            'memory': gpu_mem,
            'rtf': rtf,
            'duration': actual_dur
        })
        
        print(f"    时间: {gen_time:.2f}s | 显存: {gpu_mem:.2f}GB | RTF: {rtf:.2f}x")
    
    # 统计
    avg_time = sum(r['time'] for r in results) / len(results)
    avg_mem = sum(r['memory'] for r in results) / len(results)
    avg_rtf = sum(r['rtf'] for r in results) / len(results)
    
    print(f"\n{'='*60}")
    print(f"测试结果汇总")
    print(f"{'='*60}")
    print(f"平均生成时间: {avg_time:.2f}s")
    print(f"平均显存占用: {avg_mem:.2f}GB")
    print(f"平均实时因子: {avg_rtf:.2f}x (越小越快)")
    print(f"{'='*60}\n")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Stable Audio Open 性能测试')
    parser.add_argument('--prompt', type=str, default='ambient soundscape',
                        help='生成提示词')
    parser.add_argument('--duration', type=int, default=30,
                        help='音频时长(秒)')
    parser.add_argument('--steps', type=int, default=100,
                        help='推理步数')
    parser.add_argument('--cfg_scale', type=float, default=7,
                        help='CFG引导强度')
    parser.add_argument('--output', type=str, default='output.wav',
                        help='输出文件名')
    parser.add_argument('--benchmark', action='store_true',
                        help='运行性能测试')
    parser.add_argument('--iterations', type=int, default=5,
                        help='测试迭代次数')
    args = parser.parse_args()
    
    # 检查GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # 加载模型
    print("\n加载模型...")
    load_start = time.time()
    
    from stable_audio_tools import get_pretrained_model
    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    model = model.to(device)
    
    load_time = time.time() - load_start
    print(f"模型加载完成: {load_time:.2f}s")
    print(f"采样率: {model_config['sample_rate']}Hz")
    print(f"最大时长: {model_config['sample_size'] / model_config['sample_rate']:.1f}s")
    
    if args.benchmark:
        # 性能测试模式
        run_benchmark(model, model_config, args.iterations, args.duration, args.steps)
    else:
        # 单次生成模式
        print(f"\n生成音频: {args.prompt}")
        output, sr, gen_time, gpu_mem, actual_dur = generate_audio(
            model, model_config, args.prompt, args.duration, args.steps, args.cfg_scale
        )
        
        # 保存
        torchaudio.save(args.output, output, sr)
        print(f"\n生成完成!")
        print(f"  文件: {args.output}")
        print(f"  时长: {actual_dur:.1f}s")
        print(f"  生成时间: {gen_time:.2f}s")
        print(f"  显存占用: {gpu_mem:.2f}GB")
        print(f"  实时因子: {gen_time/actual_dur:.2f}x")

if __name__ == "__main__":
    main()
