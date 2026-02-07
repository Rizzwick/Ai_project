import os

# FORCE disable hf_transfer (Windows-safe)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"


import torch
from diffusers import StableDiffusionInpaintPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model

# --------------------------------------------------
# Device
# --------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# Load Stable Diffusion Inpainting Pipeline
# --------------------------------------------------
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float32   # REQUIRED for GTX 1650 Ti
).to(device)

# --------------------------------------------------
# Freeze base model
# --------------------------------------------------
pipe.vae.requires_grad_(False)
pipe.text_encoder.requires_grad_(False)
pipe.unet.requires_grad_(False)

# --------------------------------------------------
# Enable memory optimizations
# --------------------------------------------------
pipe.unet.enable_gradient_checkpointing()
pipe.enable_attention_slicing()

# --------------------------------------------------
# LoRA configuration (NO task_type)
# --------------------------------------------------
lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=[
        "to_q",
        "to_k",
        "to_v",
        "to_out.0"
    ],
    lora_dropout=0.05,
    bias="none"
)

# --------------------------------------------------
# Apply LoRA to UNet
# --------------------------------------------------
pipe.unet = get_peft_model(pipe.unet, lora_config)
pipe.unet.train()

# --------------------------------------------------
# Expose components
# --------------------------------------------------
unet = pipe.unet
tokenizer = pipe.tokenizer
scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
