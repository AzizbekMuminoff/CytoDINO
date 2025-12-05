import torch
import torch.nn as nn

class TransformerHead(nn.Module):
    def __init__(self, hidden_size, num_classes, num_heads=8, num_layers=2):
        super().__init__()
        self.query_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size*2,
            dropout=0.2, activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.normal_(self.query_token, std=0.02)

    def forward(self, hidden_states):
        b = hidden_states.shape[0]
        query = self.query_token.expand(b, -1, -1)
        x = self.transformer_decoder(tgt=query, memory=hidden_states)
        return self.fc(self.norm(x).squeeze(1))