import torch

CONFIG = {
    "checkpoint": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "batch_size": 1,
    "accumulation_steps": 4,
    "lr": 3e-4,
    "epochs": 8,
    "image_size": 256,
    "lora_rank": 64,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 2,
    "patience": 5,
    "smoothing": 0.1,
    "within_lineage_weight": 0.95,
    "sequential_bonus": 0.01,
    "critical_penalty": 3.0,
    "gamma": 2.0,
}
