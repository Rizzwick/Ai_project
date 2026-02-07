from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline
from peft import PeftModel

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float32
).to(device)

# Load LoRA
pipe.unet = PeftModel.from_pretrained(
    pipe.unet,
    "outputs/lora/epoch_5"
)

pipe.unet.eval()

def run_inference(
    source_path,
    mask_path,
    instruction,
    out_path="outputs/samples/result.png"
):
    source = Image.open(source_path).convert("RGB")
    mask   = Image.open(mask_path).convert("L")

    image = pipe(
        prompt=instruction,
        image=source,
        mask_image=mask,
        guidance_scale=7.5,
        num_inference_steps=25
    ).images[0]

    image.save(out_path)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    run_inference(
        "example/source.png",
        "example/mask.png",
        "replace the sky with a sunset"
    )
