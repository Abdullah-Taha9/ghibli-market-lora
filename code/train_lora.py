"""
Simple LoRA training script for Stable Diffusion 1.5
"""

import os
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import safetensors.torch as safetensors
from tqdm import tqdm
import logging

from utils.dataset import create_dataloader
from utils.config import Config


def setup_logging():
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)


def add_token(tokenizer, text_encoder, token):
    """Add new token to tokenizer and text encoder"""
    tokenizer.add_tokens(token)
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_id = tokenizer.convert_tokens_to_ids(token)
    
    # Initialize new token embedding
    with torch.no_grad():
        text_encoder.text_model.embeddings.token_embedding.weight[token_id] = \
            text_encoder.text_model.embeddings.token_embedding.weight[-2:-1].mean(dim=0)
    
    logging.info(f"Added token '{token}' with ID {token_id}")
    return token_id


def setup_lora(model, rank=8, alpha=32):
    """Setup LoRA for model"""
    if isinstance(model, UNet2DConditionModel):
        target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
        task_type = TaskType.DIFFUSION_UNET
    else:  # Text encoder
        target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
        task_type = TaskType.FEATURE_EXTRACTION
    
    lora_config = LoraConfig(
        r=rank, lora_alpha=alpha, target_modules=target_modules,
        lora_dropout=0.1, bias="none", task_type=task_type
    )
    
    return get_peft_model(model, lora_config)


def encode_prompt(text_encoder, tokenizer, prompt, device):
    """Encode text prompt"""
    inputs = tokenizer(prompt, padding="max_length", max_length=77, 
                      truncation=True, return_tensors="pt")
    return text_encoder(inputs.input_ids.to(device))[0]


def save_lora_weights(unet, text_encoder, output_path):
    """Save LoRA weights"""
    lora_state_dict = {}
    
    # UNet LoRA weights
    for key, value in unet.state_dict().items():
        if "lora_" in key:
            lora_state_dict[f"unet.{key}"] = value
    
    # Text encoder LoRA weights
    for key, value in text_encoder.state_dict().items():
        if "lora_" in key:
            lora_state_dict[f"text_encoder.{key}"] = value
    
    safetensors.save_file(lora_state_dict, output_path)
    logging.info(f"Saved LoRA weights to {output_path}")


def train():
    logger = setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create directories
    Path(Config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(Config.LOG_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load models
    logger.info("Loading models...")
    tokenizer = CLIPTokenizer.from_pretrained(Config.MODEL_NAME, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(Config.MODEL_NAME, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(Config.MODEL_NAME, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(Config.MODEL_NAME, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(Config.MODEL_NAME, subfolder="scheduler")
    
    # Move to device
    text_encoder.to(device)
    vae.to(device) 
    unet.to(device)
    vae.requires_grad_(False)
    
    # Add token
    add_token(tokenizer, text_encoder, Config.INSTANCE_TOKEN)
    
    # Setup LoRA
    logger.info("Setting up LoRA...")
    unet_lora = setup_lora(unet, Config.LORA_RANK, Config.LORA_ALPHA)
    text_encoder_lora = setup_lora(text_encoder, Config.LORA_RANK, Config.LORA_ALPHA)
    
    # Create dataloader
    dataloader = create_dataloader(Config.DATA_DIR, Config.BATCH_SIZE, 
                                 Config.INSTANCE_TOKEN, Config.RESOLUTION)
    
    # Optimizer
    params = list(unet_lora.parameters()) + list(text_encoder_lora.parameters())
    optimizer = torch.optim.AdamW(params, lr=Config.LEARNING_RATE)
    
    # Training loop
    logger.info(f"Starting training for {Config.MAX_STEPS} steps...")
    unet_lora.train()
    text_encoder_lora.train()
    
    global_step = 0
    progress_bar = tqdm(range(Config.MAX_STEPS))
    
    while global_step < Config.MAX_STEPS:
        for batch in dataloader:
            if global_step >= Config.MAX_STEPS:
                break
            
            # Get batch
            pixel_values = batch["pixel_values"].to(device)
            prompts = batch["prompts"]
            
            # Encode to latents
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
            
            # Add noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (latents.shape[0],), device=device)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Encode prompts
            prompt_embeds = []
            for prompt in prompts:
                embed = encode_prompt(text_encoder_lora, tokenizer, prompt, device)
                prompt_embeds.append(embed)
            prompt_embeds = torch.cat(prompt_embeds, dim=0)
            
            # Predict noise
            noise_pred = unet_lora(noisy_latents, timesteps, prompt_embeds).sample
            
            # Loss and backprop
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            
            if (global_step + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            global_step += 1
            progress_bar.update(1)
            
            # Log
            if global_step % Config.LOG_STEPS == 0:
                logger.info(f"Step {global_step}, Loss: {loss.item():.6f}")
            
            # Save checkpoint
            if global_step % Config.SAVE_STEPS == 0:
                checkpoint_path = Path(Config.OUTPUT_DIR) / f"checkpoint_{global_step}.safetensors"
                save_lora_weights(unet_lora, text_encoder_lora, checkpoint_path)
    
    # Save final weights
    final_path = Path(Config.OUTPUT_DIR) / "pytorch_lora_weights.safetensors"
    save_lora_weights(unet_lora, text_encoder_lora, final_path)
    logger.info("Training completed!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=Config.DATA_DIR)
    parser.add_argument("--max_steps", type=int, default=Config.MAX_STEPS)
    parser.add_argument("--learning_rate", type=float, default=Config.LEARNING_RATE)
    
    args = parser.parse_args()
    
    # Override config
    Config.DATA_DIR = args.data_dir
    Config.MAX_STEPS = args.max_steps
    Config.LEARNING_RATE = args.learning_rate
    
    train()


if __name__ == "__main__":
    main()