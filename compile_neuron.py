#!/usr/bin/env python3
"""
Stable Audio Open - Neuron 编译脚本
参考 neuronx-distributed-inference 的分阶段编译方法
"""
import os
import torch
import torch_neuronx
import argparse

LATENT_DIM = 64
SEQ_LEN = 1024  # sample_size // 2048 = 2097152 // 2048

class TimestepEmbed(torch.nn.Module):
    """时间步嵌入"""
    def __init__(self, dit):
        super().__init__()
        self.timestep_features = dit.timestep_features
        self.to_timestep_embed = dit.to_timestep_embed
    
    def forward(self, t):
        t_feat = self.timestep_features(t)
        return self.to_timestep_embed(t_feat)

class CondEmbed(torch.nn.Module):
    """条件嵌入"""
    def __init__(self, dit):
        super().__init__()
        self.to_cond_embed = dit.to_cond_embed
        self.to_global_embed = dit.to_global_embed
    
    def forward(self, cond, global_cond):
        cond_emb = self.to_cond_embed(cond)
        global_emb = self.to_global_embed(global_cond)
        return cond_emb, global_emb

class PreprocessConv(torch.nn.Module):
    """预处理卷积"""
    def __init__(self, dit):
        super().__init__()
        self.conv = dit.preprocess_conv
    
    def forward(self, x):
        return self.conv(x)

class TransformerBlock(torch.nn.Module):
    """单个 Transformer 层"""
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
    
    def forward(self, x, t_emb, cond, cond_emb, global_emb):
        return self.layer(x, t_emb, cond=cond, cond_embed=cond_emb, global_embed=global_emb)

class PostprocessConv(torch.nn.Module):
    """后处理卷积"""
    def __init__(self, dit):
        super().__init__()
        self.conv = dit.postprocess_conv
    
    def forward(self, x):
        return self.conv(x)

def compile_model(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    print("加载 stable-audio-open 模型...")
    from stable_audio_tools import get_pretrained_model
    model, cfg = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    dit = model.model.model
    dit.eval()
    
    print(f"sample_size: {cfg['sample_size']}, seq_len: {SEQ_LEN}")
    
    # 1. 编译时间步嵌入
    print("\n编译 TimestepEmbed...")
    ts_embed = TimestepEmbed(dit).eval()
    t_example = torch.ones(1)
    traced = torch_neuronx.trace(ts_embed, t_example)
    traced.save(f"{output_dir}/timestep_embed.pt")
    
    # 2. 编译预处理卷积
    print("编译 PreprocessConv...")
    preconv = PreprocessConv(dit).eval()
    x_example = torch.randn(1, LATENT_DIM, SEQ_LEN)
    traced = torch_neuronx.trace(preconv, x_example)
    traced.save(f"{output_dir}/preprocess_conv.pt")
    
    # 3. 编译 Transformer 的 project_in
    print("编译 Transformer project_in...")
    proj_in = dit.transformer.project_in.eval()
    # preprocess_conv 输出后 transpose
    x_proj = torch.randn(1, SEQ_LEN, dit.transformer.project_in.in_features)
    traced = torch_neuronx.trace(proj_in, x_proj)
    traced.save(f"{output_dir}/project_in.pt")
    
    # 4. 编译每个 Transformer 层
    print(f"编译 {len(dit.transformer.layers)} 个 Transformer 层...")
    hidden_dim = dit.transformer.project_in.out_features
    cond_dim = 768  # cross attention condition dim
    
    for i, layer in enumerate(dit.transformer.layers):
        print(f"  Layer {i}...")
        block = TransformerBlock(layer).eval()
        
        x = torch.randn(1, SEQ_LEN, hidden_dim)
        t_emb = torch.randn(1, hidden_dim)
        cond = torch.randn(1, 512, cond_dim)  # cross attention
        cond_emb = torch.randn(1, hidden_dim)
        global_emb = torch.randn(1, hidden_dim)
        
        try:
            traced = torch_neuronx.trace(
                block, (x, t_emb, cond, cond_emb, global_emb),
                compiler_args=["--model-type=transformer", "--auto-cast=all", "--auto-cast-type=bf16"]
            )
            traced.save(f"{output_dir}/layer_{i}.pt")
        except Exception as e:
            print(f"    Layer {i} 编译失败: {e}")
            break
    
    # 5. 编译 project_out
    print("编译 Transformer project_out...")
    proj_out = dit.transformer.project_out.eval()
    x_out = torch.randn(1, SEQ_LEN, hidden_dim)
    traced = torch_neuronx.trace(proj_out, x_out)
    traced.save(f"{output_dir}/project_out.pt")
    
    # 6. 编译后处理卷积
    print("编译 PostprocessConv...")
    postconv = PostprocessConv(dit).eval()
    x_post = torch.randn(1, LATENT_DIM, SEQ_LEN)
    traced = torch_neuronx.trace(postconv, x_post)
    traced.save(f"{output_dir}/postprocess_conv.pt")
    
    print(f"\n完成! 模型保存到 {output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", default="./compiled_audio")
    compile_model(parser.parse_args().output)
