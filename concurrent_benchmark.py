"""
Stable Audio Open 并发性能测试
最大化 GPU 利用率
"""
import torch
import torchaudio
import argparse
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from einops import rearrange

def get_gpu_stats():
    """获取GPU统计"""
    if torch.cuda.is_available():
        mem_used = torch.cuda.memory_allocated() / 1024**3
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return mem_used, mem_total
    return 0, 0

def generate_single(model, model_config, prompt, duration, steps, device, batch_idx):
    """单次生成"""
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
    
    start = time.time()
    output = generate_diffusion_cond(
        model,
        steps=steps,
        cfg_scale=7,
        conditioning=conditioning,
        sample_size=sample_size,
        sigma_min=0.3,
        sigma_max=500,
        sampler_type="dpmpp-3m-sde",
        device=device
    )
    gen_time = time.time() - start
    
    return batch_idx, gen_time, actual_duration

def test_batch_size(model, model_config, batch_size, duration, steps, device):
    """测试不同 batch size 的性能"""
    from stable_audio_tools.inference.generation import generate_diffusion_cond
    
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    max_duration = sample_size / sample_rate
    actual_duration = min(duration, max_duration)
    
    # 构建 batch conditioning
    prompts = [
        "128 BPM tech house drum loop",
        "ambient soundscape with rain",
        "cinematic orchestral music",
        "electronic bass drop",
        "acoustic guitar melody",
        "piano jazz improvisation",
        "heavy metal guitar riff",
        "lo-fi hip hop beat"
    ]
    
    conditioning = []
    for i in range(batch_size):
        conditioning.append({
            "prompt": prompts[i % len(prompts)],
            "seconds_start": 0,
            "seconds_total": actual_duration
        })
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.time()
    
    try:
        output = generate_diffusion_cond(
            model,
            steps=steps,
            cfg_scale=7,
            conditioning=conditioning,
            sample_size=sample_size,
            sigma_min=0.3,
            sigma_max=500,
            sampler_type="dpmpp-3m-sde",
            device=device
        )
        torch.cuda.synchronize()
        total_time = time.time() - start
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        
        return {
            'batch_size': batch_size,
            'success': True,
            'total_time': total_time,
            'time_per_audio': total_time / batch_size,
            'peak_memory': peak_mem,
            'throughput': batch_size * actual_duration / total_time,  # 秒音频/秒
            'duration': actual_duration
        }
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {
            'batch_size': batch_size,
            'success': False,
            'error': 'OOM'
        }

def find_optimal_batch(model, model_config, duration, steps, device):
    """找到最优 batch size"""
    print("\n" + "="*70)
    print("寻找最优 Batch Size (最大化吞吐量)")
    print("="*70)
    
    results = []
    batch_sizes = [1, 2, 3, 4, 6, 8]
    
    for bs in batch_sizes:
        print(f"\n测试 batch_size={bs}...")
        result = test_batch_size(model, model_config, bs, duration, steps, device)
        
        if result['success']:
            print(f"  ✓ 总时间: {result['total_time']:.2f}s | "
                  f"单条: {result['time_per_audio']:.2f}s | "
                  f"显存: {result['peak_memory']:.2f}GB | "
                  f"吞吐: {result['throughput']:.2f}x")
            results.append(result)
        else:
            print(f"  ✗ OOM - 显存不足")
            break
    
    if results:
        best = max(results, key=lambda x: x['throughput'])
        print("\n" + "="*70)
        print(f"最优配置: batch_size={best['batch_size']}")
        print(f"  - 吞吐量: {best['throughput']:.2f}x 实时")
        print(f"  - 单条音频: {best['time_per_audio']:.2f}s")
        print(f"  - 显存占用: {best['peak_memory']:.2f}GB")
        print("="*70)
        return best
    return None

def concurrent_stress_test(model, model_config, num_requests, batch_size, duration, steps, device):
    """并发压力测试"""
    print("\n" + "="*70)
    print(f"并发压力测试: {num_requests} 请求, batch_size={batch_size}")
    print("="*70)
    
    total_start = time.time()
    completed = 0
    total_audio_seconds = 0
    
    for i in range(0, num_requests, batch_size):
        current_batch = min(batch_size, num_requests - i)
        result = test_batch_size(model, model_config, current_batch, duration, steps, device)
        
        if result['success']:
            completed += current_batch
            total_audio_seconds += current_batch * result['duration']
            print(f"  批次 {i//batch_size + 1}: {current_batch} 条完成, "
                  f"耗时 {result['total_time']:.2f}s")
    
    total_time = time.time() - total_start
    
    print("\n" + "-"*70)
    print(f"压力测试结果:")
    print(f"  - 完成请求: {completed}/{num_requests}")
    print(f"  - 总耗时: {total_time:.2f}s")
    print(f"  - 生成音频: {total_audio_seconds:.1f}s")
    print(f"  - 整体吞吐: {total_audio_seconds/total_time:.2f}x 实时")
    print(f"  - QPS: {completed/total_time:.2f} 请求/秒")
    print("="*70)

def main():
    parser = argparse.ArgumentParser(description='Stable Audio 并发性能测试')
    parser.add_argument('--duration', type=int, default=30, help='音频时长(秒)')
    parser.add_argument('--steps', type=int, default=100, help='推理步数')
    parser.add_argument('--find-optimal', action='store_true', help='寻找最优batch size')
    parser.add_argument('--batch-size', type=int, default=0, help='指定batch size测试')
    parser.add_argument('--stress', type=int, default=0, help='压力测试请求数')
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}")
        print(f"显存: {props.total_memory / 1024**3:.1f}GB")
    
    print("\n加载模型...")
    load_start = time.time()
    
    from stable_audio_tools import get_pretrained_model
    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    model = model.to(device)
    
    print(f"模型加载完成: {time.time() - load_start:.2f}s")
    
    if args.find_optimal:
        find_optimal_batch(model, model_config, args.duration, args.steps, device)
    
    if args.batch_size > 0:
        print(f"\n测试 batch_size={args.batch_size}")
        result = test_batch_size(model, model_config, args.batch_size, args.duration, args.steps, device)
        if result['success']:
            print(f"  总时间: {result['total_time']:.2f}s")
            print(f"  单条: {result['time_per_audio']:.2f}s")
            print(f"  显存: {result['peak_memory']:.2f}GB")
            print(f"  吞吐: {result['throughput']:.2f}x 实时")
    
    if args.stress > 0:
        # 先找最优 batch size
        best = find_optimal_batch(model, model_config, args.duration, args.steps, device)
        if best:
            concurrent_stress_test(model, model_config, args.stress, 
                                   best['batch_size'], args.duration, args.steps, device)

if __name__ == "__main__":
    main()
