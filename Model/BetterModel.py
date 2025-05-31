import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy



class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 512, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(-torch.arange(0, d_model, 2, dtype=torch.float) * (torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)  @property
    def total_parameters(self) -> int:
        #liczy liczbę parametrów:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

#
# model = ChessTransformer(d_model=256, num_heads=4, num_layers=4)
# print(f"Total parameters: {model.total_parameters}")
