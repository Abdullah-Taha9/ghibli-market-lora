import torch
import torch.nn as nn
import safetensors.torch as safetensors

# ---- LoRA module wrapper ----
class LoRALinear(nn.Module):
    def __init__(self, base_layer, lora_A, lora_B, alpha=1.0):
        super().__init__()
        self.base = base_layer

        # register as parameters (same dtype/device as base layer)
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        self.lora_A = nn.Parameter(lora_A.to(device=device, dtype=dtype))
        self.lora_B = nn.Parameter(lora_B.to(device=device, dtype=dtype))

        self.scaling = alpha / lora_A.size(0)

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling



def apply_lora_to_unet_and_text(pipe, lora_path):
    state_dict = safetensors.load_file(lora_path)

    # ---- Patch UNet ----
    for name, module in pipe.unet.named_modules():
        if isinstance(module, nn.Linear):
            # check if LoRA weights exist for this module
            A_key = f"unet.{name}.lora_A.default.weight"
            B_key = f"unet.{name}.lora_B.default.weight"
            if A_key in state_dict and B_key in state_dict:
                print(f"Applying LoRA to UNet layer: {name}")
                lora_A = state_dict[A_key].to(module.weight.device, module.weight.dtype)
                lora_B = state_dict[B_key].to(module.weight.device, module.weight.dtype)
                alpha = lora_A.size(0)  # usually rank
                new_layer = LoRALinear(module, lora_A, lora_B, alpha=alpha)
                # replace module with wrapped version
                parent = pipe.unet
                for attr in name.split(".")[:-1]:
                    parent = getattr(parent, attr)
                setattr(parent, name.split(".")[-1], new_layer)

    # ---- Patch Text Encoder ----
    for name, module in pipe.text_encoder.named_modules():
        if isinstance(module, nn.Linear):
            A_key = f"text_encoder.{name}.lora_A.default.weight"
            B_key = f"text_encoder.{name}.lora_B.default.weight"
            if A_key in state_dict and B_key in state_dict:
                print(f"Applying LoRA to TextEncoder layer: {name}")
                lora_A = state_dict[A_key]
                lora_B = state_dict[B_key]
                alpha = lora_A.size(0)
                new_layer = LoRALinear(module, lora_A, lora_B, alpha=alpha)
                parent = pipe.text_encoder
                for attr in name.split(".")[:-1]:
                    parent = getattr(parent, attr)
                setattr(parent, name.split(".")[-1], new_layer)

    print("✅ LoRA adapters applied!")
    return pipe

from diffusers import StableDiffusionPipeline
import torch

MODEL_PATH = "/home/hpc/v123be/v123be39/dl/sd15"
LORA_PATH = "lora_out/pytorch_lora_weights.safetensors"

# Load base pipeline
pipe = StableDiffusionPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to("cuda")

# Apply LoRA adapters
pipe = apply_lora_to_unet_and_text(pipe, LORA_PATH)

# Generate
prompt = "a car, in <sks> style"
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("sample_lora_applied.png")
