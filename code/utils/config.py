# Model and tokenizer
MODEL_NAME = "stable-diffusion-v1-5/stable-diffusion-v1-5"
INSTANCE_TOKEN = "<sks>"

# Training parameters  
RESOLUTION = 512
LORA_RANK = 8
LEARNING_RATE = 1e-4
MAX_STEPS = 800
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4

# Paths
DATA_DIR = "data/512"
OUTPUT_DIR = "lora_out"
SAMPLES_DIR = "samples"
