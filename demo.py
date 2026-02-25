import torch
import torchaudio
import time
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

device = "cuda"

# ===== 优化设置 =====
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

# ===== 加载模型 =====
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]

model = model.to(device)
model.eval()

# ===== 只对 diffusion backbone 用 bfloat16 =====
if hasattr(model, 'model') and hasattr(model.model, 'model'):
    model.model.model = model.model.model.to(dtype=torch.bfloat16)
    print("✅ Diffusion backbone -> bfloat16")

prompt = "relaxing casual puzzle game background music, gentle melodic patterns, soft synth layers, calm rhythmic pulses, light playful chimes and mellow tones, immersive and soothing atmosphere, matching the gentle breathing glow and slow ping pong motion, high quality, clear and engaging, perfect for a peaceful and focused puzzle environment"

conditioning = [{
    "prompt": prompt,
    "seconds_start": 0,
    "seconds_total": 30
}]

# ===== 推理 =====
torch.cuda.synchronize()
start = time.time()

with torch.inference_mode():
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
print(f"⏱️推理耗时: {time.time() - start:.2f}s")

# ===== 保存 =====
output = rearrange(output, "b d n -> d (b n)")
output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
torchaudio.save("/tmp/output.wav", output, sample_rate)
print("✅完成")
