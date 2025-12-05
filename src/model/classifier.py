import torch.nn as nn
from .backbone import DinoBackbone
from .transformer_decoder import TransformerHead

class DinoV3Learner(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = DinoBackbone()
        self.head = TransformerHead(self.backbone.hidden_size, num_classes, num_heads=4, num_layers=4)

    def forward(self, pixel_values):
        hidden_states = self.backbone(pixel_values)
        return self.head(hidden_states)