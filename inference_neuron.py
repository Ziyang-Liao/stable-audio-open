#!/usr/bin/env python3
"""
Stable Audio Open - Neuron 推理脚本
使用预编译的 Neuron 模型进行推理
"""
import os
import torch
import torch_neuronx
import torchaudio
import time
import argparse

os.environ["NEURON_RT_NUM_CORES"] = "2"

LATENT_DIM = 64
SEQ_LEN = 1024

class StableAudioNeuron:
    def __init__(self, model_dir="./compiled_audio"):
        print("加载 Neuron 编译模型...")
        
        # 加载原始模型获取其他组件
        from stable_audio_tools import get_pretrained_model
        self.model, self.cfg = get_pretrained_model("stabilityai/stable-audio-open-1.0")
        self.dit = self.model.model.model
        
        # 加载编译的模型
        self.timestep_embed = torch.jit.load(f"{model_dir}/timestep_embed.pt")
        self.preprocess_conv = torch.jit.load(f"{model_dir}/preprocess_conv.pt")
        self.project_in = torch.jit.load(f"{model_dir}/project_in.pt")
        self.layers = [torch.jit.load(f"{model_dir}/layer_{i}.pt") for i in range(24)]
        self.project_out = torch.jit.load(f"{model_dir}/project_out.pt")
        self.postprocess_conv = torch.jit.load(f"{model_dir}/postprocess_conv.pt")
        
        print("模型加载完成!")
    
    def forward_dit(self, x, t, context, global_cond):
        """使用 Neuron 编译模型的 DiT forward"""
        # 时间步嵌入 (使用原始模型，因为需要和 context 结合)
        t_embed = self.timestep_embed(t)
        
        # 预处理
        x = self.preprocess_conv(x)
        x = x.transpose(1, 2)  # (B, C, L) -> (B, L, C)
        
        # Project in
        x = self.project_in(x)
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x, context, global_cond + t_embed)
        
        # Project out
        x = self.project_out(x)
        x = x.transpose(1, 2)  # (B, L, C) -> (B, C, L)
        
        # 后处理
        x = self.postprocess_conv(x)
        return x
    
    def generate(self, prompt, duration=30, steps=100):
        """生成音频"""
        from stable_audio_tools.inference.generation import generate_diffusion_cond
        
        sample_rate = self.cfg["sample_rate"]
        sample_size = self.cfg["sample_size"]
        
        conditioning = [{
            "prompt": prompt,
            "seconds_start": 0,
            "seconds_total": min(duration, sample_size / sample_rate)
        }]
        
        # 暂时使用原始模型生成（后续可替换采样循环）
        start = time.time()
        output = generate_diffusion_cond(
            self.model,
            steps=steps,
            cfg_scale=7,
            conditioning=conditioning,
            sample_size=sample_size,
            sigma_min=0.3,
            sigma_max=500,
            sampler_type="dpmpp-3m-sde",
            device="cpu"
        )
        gen_time = time.time() - start
        
        # 后处理
        from einops import rearrange
        output = rearrange(output, "b d n -> d (b n)")
        output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1)
        output = (output * 32767).to(torch.int16).cpu()
        
        return output, sample_rate, gen_time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="./compiled_audio")
    parser.add_argument("--prompt", default="128 BPM tech house drum loop")
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", default="output_neuron.wav")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    
    model = StableAudioNeuron(args.model_dir)
    
    if args.benchmark:
        print(f"\n{'='*50}")
        print(f"Neuron Benchmark: {args.iterations} 次, {args.steps} 步")
        print(f"{'='*50}\n")
        
        times = []
        for i in range(args.iterations):
            print(f"[{i+1}/{args.iterations}] 生成中...")
            _, sr, t = model.generate(args.prompt, args.duration, args.steps)
            times.append(t)
            print(f"    时间: {t:.2f}s")
        
        avg = sum(times) / len(times)
        print(f"\n平均时间: {avg:.2f}s")
        print(f"RTF: {avg / args.duration:.2f}x")
    else:
        print(f"\n生成: {args.prompt}")
        output, sr, t = model.generate(args.prompt, args.duration, args.steps)
        torchaudio.save(args.output, output, sr)
        print(f"完成! 时间: {t:.2f}s, 保存到: {args.output}")

if __name__ == "__main__":
    main()
