"""
Stable Audio Open 并发性能测试
基于官方示例代码，测试多进程并行
"""
import torch
import torchaudio
import time
import os
import argparse
from multiprocessing import Process, Queue
from einops import rearrange

def single_generate(model, model_config, prompt, duration, device):
    """单次生成"""
    from stable_audio_tools.inference.generation import generate_diffusion_cond
    
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    
    conditioning = [{
        "prompt": prompt,
        "seconds_start": 0,
        "seconds_total": duration
    }]
    
    output = generate_diffusion_cond(
        model,
        steps=100,
        cfg_scale=7,
        conditioning=conditioning,
        sample_size=sample_size,
        sigma_min=0.3,
        sigma_max=500,
        sampler_type="dpmpp-3m-sde",
        device=device
    )
    
    output = rearrange(output, "b d n -> d (b n)")
    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    
    return output, sample_rate

def worker_process(worker_id, prompt, duration, result_queue):
    """独立进程 worker"""
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    import torch
    from stable_audio_tools import get_pretrained_model
    from stable_audio_tools.inference.generation import generate_diffusion_cond
    from einops import rearrange
    
    device = "cuda"
    
    # 每个进程独立加载模型
    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    model = model.to(device)
    
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    
    conditioning = [{
        "prompt": prompt,
        "seconds_start": 0,
        "seconds_total": duration
    }]
    
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    
    output = generate_diffusion_cond(
        model,
        steps=100,
        cfg_scale=7,
        conditioning=conditioning,
        sample_size=sample_size,
        sigma_min=0.3,
        sigma_max=500,
        sampler_type="dpmpp-3m-sde",
        device=device
    )
    
    torch.cuda.synchronize()
    gen_time = time.time() - start
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    
    result_queue.put({
        'worker_id': worker_id,
        'time': gen_time,
        'memory': peak_mem,
        'duration': duration
    })

def test_parallel(num_workers, duration=30):
    """测试多进程并行"""
    prompts = [
        "128 BPM tech house drum loop",
        "ambient soundscape with rain",
        "cinematic orchestral music",
        "electronic bass drop",
    ]
    
    print(f"\n{'='*60}")
    print(f"并行测试: {num_workers} 个进程同时运行")
    print(f"{'='*60}")
    
    result_queue = Queue()
    processes = []
    
    start_time = time.time()
    
    # 启动所有进程
    for i in range(num_workers):
        p = Process(target=worker_process, args=(i, prompts[i % len(prompts)], duration, result_queue))
        p.start()
        processes.append(p)
        print(f"  启动 Worker {i}")
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    total_time = time.time() - start_time
    
    # 收集结果
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    
    print(f"\n{'-'*60}")
    print(f"各进程结果:")
    total_audio = 0
    max_mem = 0
    for r in sorted(results, key=lambda x: x['worker_id']):
        print(f"  Worker {r['worker_id']}: {r['time']:.2f}s | 显存: {r['memory']:.2f}GB")
        total_audio += r['duration']
        max_mem = max(max_mem, r['memory'])
    
    print(f"\n{'-'*60}")
    print(f"汇总:")
    print(f"  并行进程数: {num_workers}")
    print(f"  总耗时: {total_time:.2f}s")
    print(f"  生成音频: {total_audio:.0f}s ({num_workers} x {duration}s)")
    print(f"  吞吐量: {total_audio/total_time:.2f}x 实时")
    print(f"  峰值显存: {max_mem:.2f}GB")
    print(f"{'='*60}")
    
    return total_time, total_audio, max_mem

def test_sequential(num_requests, duration=30):
    """顺序执行测试"""
    from stable_audio_tools import get_pretrained_model
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*60}")
    print(f"顺序执行测试: {num_requests} 请求")
    print(f"{'='*60}")
    
    print("加载模型...")
    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    model = model.to(device)
    print("模型加载完成")
    
    prompts = [
        "128 BPM tech house drum loop",
        "ambient soundscape with rain",
        "cinematic orchestral music",
        "electronic bass drop",
    ]
    
    total_start = time.time()
    times = []
    
    for i in range(num_requests):
        prompt = prompts[i % len(prompts)]
        print(f"[{i+1}/{num_requests}] {prompt[:30]}...", end=" ", flush=True)
        
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        output, sr = single_generate(model, model_config, prompt, duration, device)
        torch.cuda.synchronize()
        gen_time = time.time() - start
        times.append(gen_time)
        
        mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"{gen_time:.2f}s | {mem:.2f}GB")
    
    total_time = time.time() - total_start
    total_audio = num_requests * duration
    
    print(f"\n{'-'*60}")
    print(f"顺序执行结果:")
    print(f"  请求数: {num_requests}")
    print(f"  总耗时: {total_time:.2f}s")
    print(f"  平均单条: {sum(times)/len(times):.2f}s")
    print(f"  生成音频: {total_audio}s")
    print(f"  吞吐量: {total_audio/total_time:.2f}x 实时")
    print(f"{'='*60}")
    
    return total_time, total_audio

def main():
    parser = argparse.ArgumentParser(description='Stable Audio 并发测试')
    parser.add_argument('--sequential', type=int, default=0, help='顺序执行N个请求')
    parser.add_argument('--parallel', type=int, default=0, help='并行N个进程')
    parser.add_argument('--duration', type=int, default=30, help='音频时长')
    parser.add_argument('--compare', action='store_true', help='对比顺序vs并行')
    args = parser.parse_args()
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    if args.sequential > 0:
        test_sequential(args.sequential, args.duration)
    
    if args.parallel > 0:
        test_parallel(args.parallel, args.duration)
    
    if args.compare:
        print("\n" + "="*60)
        print("对比测试: 顺序 vs 并行")
        print("="*60)
        
        # 顺序执行 2 个
        seq_time, seq_audio = test_sequential(2, args.duration)
        
        # 并行执行 2 个
        par_time, par_audio, par_mem = test_parallel(2, args.duration)
        
        print(f"\n{'='*60}")
        print(f"对比结果:")
        print(f"  顺序 2 请求: {seq_time:.2f}s | 吞吐: {seq_audio/seq_time:.2f}x")
        print(f"  并行 2 进程: {par_time:.2f}s | 吞吐: {par_audio/par_time:.2f}x")
        print(f"  加速比: {seq_time/par_time:.2f}x")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()
