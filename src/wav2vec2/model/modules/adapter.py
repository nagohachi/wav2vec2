import torch
from torch import nn

from .schemas import Wav2Vec2AdapterConfig


class Wav2Vec2Adapter(nn.Module):
    def __init__(self, config: Wav2Vec2AdapterConfig) -> None:
        super().__init__()
        self.linear = nn.Linear(config.in_dim, config.out_dim)

    def forward(
        self, x: torch.Tensor, xlens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.linear(x), xlens
