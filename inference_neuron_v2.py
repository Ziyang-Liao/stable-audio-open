#!/usr/bin/env python3
"""
Stable Audio Open - Neuron 原生推理
直接使用编译的 Neuron 模型进行采样，不依赖原始采样器
"""
import os
import torch
import torch_neuronx
import torchaudio
import time
import argparse
import math

os.environ["NEURON_RT_NUM_CORES"] = "2"

class StableAudioNeuron:
    def __init__(self, model_dir="./compiled_audio"):
        print("加载模型...")
        
        from stable_audio_tools import get_pretrained_model
        self.orig_model, self.cfg = get_pretrained_model("stabilityai/stable-audio-open-1.0")
        self.dit = self.orig_model.model.model
        self.conditioner = self.orig_model.conditioner
        self.pretransform = self.orig_model.pretransform
        
        # 加载 Neuron 编译模型
        self.ts_embed = torch.jit.load(f"{model_dir}/timestep_embed.pt")
        self.pre_conv = torch.jit.load(f"{model_dir}/preprocess_conv.pt")
        self.proj_in = torch.jit.load(f"{model_dir}/project_in.pt")
        self.layers = [torch.jit.load(f"{model_dir}/layer_{i}.pt") for i in range(24)]
        self.proj_out = torch.jit.load(f"{model_dir}/project_out.pt")
        self.post_conv = torch.jit.load(f"{model_dir}/postprocess_conv.pt")
        
        # 缓存原始模型的嵌入层
        self.cond_embed = self.dit.to_cond_embed
        self.global_embed = self.dit.to_global_embed
        
        print("就绪!")
    
    def denoise(self, x, sigma, cond_embed, cross_attn):
        """单步去噪 - 使用 Neuron 模型"""
        t = sigma.view(1)
        t_emb = self.ts_embed(t)
        global_cond = cond_embed + t_emb
        
        h = self.pre_conv(x)
        h = h.transpose(1, 2)
        h = self.proj_in(h)
        
        for layer in self.layers:
            h = layer(h, cross_attn, global_cond)
        
        h = self.proj_out(h)
        h = h.transpose(1, 2)
        return self.post_conv(h)
    
    def denoise_batch(self, x, sigma, cond_embed, cross_attn):
        """Batch 去噪 (batch=2 for CFG)"""
        t = sigma.view(1)
        t_emb = self.ts_embed(t)
        
        # 分别处理 batch 中的每个样本（因为编译时 batch=1）
        results = []
        for i in range(x.shape[0]):
            global_cond = cond_embed[i:i+1] + t_emb
            h = self.pre_conv(x[i:i+1])
            h = h.transpose(1, 2)
            h = self.proj_in(h)
            
            for layer in self.layers:
                h = layer(h, cross_attn[i:i+1], global_cond)
            
            h = self.proj_out(h)
            h = h.transpose(1, 2)
            results.append(self.post_conv(h))
        
        return torch.cat(results)
    
    def sample_euler(self, shape, cond, steps=100, cfg_scale=7.0):
        """Euler 采样器"""
        # sigma 调度 (karras)
        sigma_min, sigma_max = 0.3, 500
        rho = 7
        ramp = torch.linspace(0, 1, steps)
        sigmas = (sigma_max ** (1/rho) + ramp * (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho
        sigmas = torch.cat([sigmas, torch.zeros(1)])
        
        # 初始噪声
        x = torch.randn(shape) * sigmas[0]
        
        # 条件嵌入 - prompt 是 (cross_attn, mask) tuple
        cross_attn = cond["prompt"][0]  # (1, 128, 768)
        # pad to 512 (编译时的 shape)
        if cross_attn.shape[1] < 512:
            pad = torch.zeros(1, 512 - cross_attn.shape[1], 768)
            cross_attn = torch.cat([cross_attn, pad], dim=1)
        
        # seconds_start 和 seconds_total 用于 global conditioning
        global_cond_input = torch.cat([
            cond["seconds_start"][0],
            cond["seconds_total"][0]
        ], dim=-1) if "seconds_start" in cond else torch.zeros(1, 768)
        
        cond_embed = self.global_embed(global_cond_input)
        uncond_embed = torch.zeros_like(cond_embed)
        
        print(f"采样 {steps} 步...")
        start = time.time()
        
        for i in range(steps):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            
            # CFG: batch 条件 + 无条件 一起跑
            x_in = torch.cat([x, x])
            cross_in = torch.cat([cross_attn, torch.zeros_like(cross_attn)])
            global_in = torch.cat([cond_embed, uncond_embed])
            
            # 单次 forward (batch=2)
            denoised = self.denoise_batch(x_in, sigma, global_in, cross_in)
            
            # CFG 组合
            d_cond, d_uncond = denoised.chunk(2)
            d = d_uncond + cfg_scale * (d_cond - d_uncond)
            
            # Euler 步进
            dt = sigma_next - sigma
            x = x + d * dt
            
            if (i + 1) % 20 == 0:
                print(f"  Step {i+1}/{steps}")
        
        print(f"采样完成: {time.time()-start:.2f}s")
        return x
    
    def generate(self, prompt, duration=30, steps=100):
        """生成音频"""
        sr = self.cfg["sample_rate"]
        sample_size = self.cfg["sample_size"]
        
        # 获取条件
        cond_input = {"prompt": prompt, "seconds_start": 0, "seconds_total": duration}
        cond = self.conditioner([cond_input], device="cpu")
        
        # latent shape
        latent_len = sample_size // 2048
        shape = (1, 64, latent_len)
        
        # 采样
        start = time.time()
        latent = self.sample_euler(shape, cond, steps)
        
        # 解码
        audio = self.pretransform.decode(latent)
        gen_time = time.time() - start
        
        # 后处理
        audio = audio.squeeze(0)
        audio = audio / audio.abs().max()
        audio = (audio * 32767).to(torch.int16)
        
        return audio, sr, gen_time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="./compiled_audio")
    parser.add_argument("--prompt", default="128 BPM tech house drum loop")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", default="output.wav")
    args = parser.parse_args()
    
    model = StableAudioNeuron(args.model_dir)
    
    print(f"\n生成: {args.prompt}")
    audio, sr, t = model.generate(args.prompt, steps=args.steps)
    
    # 保存 wav
    import scipy.io.wavfile as wav
    wav.write(args.output, sr, audio.numpy().T)
    print(f"\n完成! 时间: {t:.2f}s ({t/30:.2f}x RTF)")

if __name__ == "__main__":
    main()
