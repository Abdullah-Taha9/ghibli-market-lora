"""
Simple configuration for LoRA fine-tuning
"""

class Config:
    # Model settings
    MODEL_NAME = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    INSTANCE_TOKEN = "<sks>"
    
    # Training parameters
    RESOLUTION = 512
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 4
    LEARNING_RATE = 1e-4
    MAX_STEPS = 800
    
    # LoRA settings
    LORA_RANK = 8
    LORA_ALPHA = 32 # typically is 2-4x lora-rank
    LORA_DROPOUT = 0.1
    
    # Paths
    DATA_DIR = "data/512"
    OUTPUT_DIR = "lora_out"
    SAMPLES_DIR = "samples"
    LOG_DIR = "logs"
    
    # Training settings
    SAVE_STEPS = 200
    LOG_STEPS = 50
    
    # Dataset settings
    CENTER_CROP = True
    SHUFFLE_DATASET = True
    NUM_WORKERS = 0
    PIN_MEMORY = True