import torch
import time
import multiprocessing as mp
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond


def run_inference(proc_id, barrier, result_dict):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    model, model_config = get_pretrained_model('stabilityai/stable-audio-open-1.0')
    sample_size = model_config['sample_size']
    model = model.to('cuda')
    model.eval()
    if hasattr(model, 'model') and hasattr(model.model, 'model'):
        model.model.model = model.model.model.to(dtype=torch.bfloat16)

    conditioning = [{'prompt': 'relaxing puzzle game background music, gentle melodic patterns', 'seconds_start': 0, 'seconds_total': 30}]

    barrier.wait()
    torch.cuda.synchronize()
    start = time.time()
    with torch.inference_mode():
        generate_diffusion_cond(model, steps=100, cfg_scale=7, conditioning=conditioning, sample_size=sample_size, sigma_min=0.3, sigma_max=500, sampler_type='dpmpp-3m-sde', device='cuda')
    torch.cuda.synchronize()
    elapsed = time.time() - start
    result_dict[proc_id] = elapsed


if __name__ == '__main__':
    mp.set_start_method('spawn')
    for n in [1, 2, 3]:
        print(f'\n=== {n} 并发进程 (bfloat16) ===')
        manager = mp.Manager()
        result_dict = manager.dict()
        barrier = mp.Barrier(n)
        procs = [mp.Process(target=run_inference, args=(i, barrier, result_dict)) for i in range(n)]
        wall_start = time.time()
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        wall_time = time.time() - wall_start
        times = [result_dict[i] for i in range(n)]
        print(f'各进程耗时: {["{:.2f}s".format(t) for t in times]}')
        print(f'总墙钟时间: {wall_time:.2f}s')
        print(f'总生成音频: {n*30}s')
        print(f'吞吐量: {n*30/wall_time:.2f}x 实时')
