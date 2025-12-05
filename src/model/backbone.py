import torch.nn as nn
from transformers import AutoModel
from peft import LoraConfig, get_peft_model
from configs import CONFIG

class DinoBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        print(f"\n Loading: {CONFIG['checkpoint']}")
        self.model = AutoModel.from_pretrained(CONFIG['checkpoint'])
        for p in self.model.parameters():
            p.requires_grad = False

        peft_config = LoraConfig(
            r=CONFIG['lora_rank'], lora_alpha=128,
            target_modules="all-linear", lora_dropout=0.1, bias="none"
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        self.hidden_size = self.model.config.hidden_size

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).last_hidden_state