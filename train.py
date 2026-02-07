import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset

from dataset import MagicBrushDataset
from model import pipe, unet, tokenizer, scheduler, device


# -------------------------
# Helpers
# -------------------------

@torch.no_grad()
def encode_text(prompts):
    tokens = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt"
    ).input_ids.to(device)

    return pipe.text_encoder(tokens)[0]


@torch.no_grad()
def encode_latents(images):
    latents = pipe.vae.encode(images).latent_dist.sample()
    return latents * 0.18215


# -------------------------
# Training
# -------------------------

def train(
    epochs=5,
    batch_size=1,
    lr=1e-4,
    max_samples=2000
):
    # 🔥 LIMIT DATASET SIZE HERE
    hf_data = load_dataset(
        "osunlp/MagicBrush",
        split=f"train[:{max_samples}]"
    )

    dataset = MagicBrushDataset(hf_data)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=lr
    )

    unet.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for batch in loader:
            src  = batch["source"].to(device)   # [B,3,H,W]
            tgt  = batch["target"].to(device)   # [B,3,H,W]
            mask = batch["mask"].to(device)     # [B,1,H,W]
            text = batch["text"]

            # -------------------------
            # Encode text & images
            # -------------------------
            text_emb = encode_text(text)
            z_src = encode_latents(src)
            z_tgt = encode_latents(tgt)

            # Resize mask to latent resolution
            mask_latent = F.interpolate(
                mask,
                size=z_src.shape[-2:],
                mode="nearest"
            )

            # -------------------------
            # Diffusion step
            # -------------------------
            noise = torch.randn_like(z_tgt)

            timesteps = torch.randint(
                0,
                scheduler.config.num_train_timesteps,
                (z_tgt.size(0),),
                device=device
            ).long()

            z_noisy = scheduler.add_noise(
                z_tgt,
                noise,
                timesteps
            )

            # -------------------------
            # Inpainting UNet input
            # channels = 4 (latent) + 1 (mask) + 4 (masked image)
            # -------------------------
            masked_latent = z_src * (1 - mask_latent)

            model_input = torch.cat(
                [z_noisy, mask_latent, masked_latent],
                dim=1
            )

            # -------------------------
            # Predict noise
            # -------------------------
            noise_pred = unet(
                model_input,
                timesteps,
                encoder_hidden_states=text_emb
            ).sample

            # -------------------------
            # Masked MSE loss
            # -------------------------
            loss = F.mse_loss(
                noise_pred * mask_latent,
                noise * mask_latent
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

        # Save ONLY LoRA weights
        unet.save_pretrained(
            f"outputs/lora/epoch_{epoch+1}"
        )


if __name__ == "__main__":
    train()
