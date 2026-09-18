import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from torch import nn

from wav2vec2.model.modules.schemas import (
    Wav2Vec2QuantizationConfig,
)


class Wav2Vec2Quantization(nn.Module):
    def __init__(self, config: Wav2Vec2QuantizationConfig) -> None:
        # https://jonathanbgn.com/2021/09/30/illustrated-wav2vec-2.html
        super().__init__()
        self.config = config
        self.codebooks = nn.Parameter(
            torch.empty(
                config.num_code_group,
                config.codebook_size_per_group,
                config.concat_codebook_dim // config.num_code_group,
            ).uniform_()
        )  # (G, V, d/G)

        self.quantization_linear_in = nn.Linear(
            config.quantization_in_dim,
            config.num_code_group * config.codebook_size_per_group,
        )
        self.quantization_proj_out = nn.Linear(
            config.concat_codebook_dim, config.quantization_out_dim
        )

    def _lens_to_valid_mask(self, xlens: torch.Tensor, max_len: int) -> torch.Tensor:
        positions = torch.arange(max_len, device=xlens.device)
        positions_2d = rearrange(positions, "s -> 1 s")
        xlens_2d = rearrange(xlens, "b -> b 1")
        return xlens_2d > positions_2d

    def forward(self, x: torch.Tensor, xlens: torch.Tensor):
        x = self.quantization_linear_in(x)

        x_logits_groupby = rearrange(
            x, "b s (G V) -> b s G V", G=self.config.num_code_group
        )
        one_hot_x = F.gumbel_softmax(x_logits_groupby, hard=True, dim=-1)

        codebooks_selected = einsum(
            self.codebooks, one_hot_x, "G V d_G, b s G V -> b s G d_G"
        )
        codebooks_concat = rearrange(codebooks_selected, "b s G d_G -> b s (G d_G)")

        # diversity loss
        x_prob_groupby = F.softmax(x_logits_groupby, dim=-1)

        # todo: mask -> sum over b, s -> (G, V) -> sum p log p over V dim -> sum over G -> * 1/GV
        # diversity_loss = ...

        return self.quantization_proj_out(codebooks_concat), xlens
