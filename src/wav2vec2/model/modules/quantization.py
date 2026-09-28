import torch
import torch.nn.functional as F
from einops import einsum, rearrange, reduce
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

    @staticmethod
    def _lens_to_valid_mask(xlens: torch.Tensor, max_len: int) -> torch.Tensor:
        positions = torch.arange(max_len, device=xlens.device)
        positions_2d = rearrange(positions, "s -> 1 s")
        xlens_2d = rearrange(xlens, "b -> b 1")
        return xlens_2d > positions_2d

    def _logits_group_by_codebook(self, x_logits: torch.Tensor) -> torch.Tensor:
        return rearrange(x_logits, "b s (G V) -> b s G V", G=self.config.num_code_group)

    def _select_codebooks(self, x_logits: torch.Tensor) -> torch.Tensor:
        x_logits_groupby = self._logits_group_by_codebook(x_logits)
        one_hot_x = F.gumbel_softmax(x_logits_groupby, hard=True, dim=-1)
        return einsum(self.codebooks, one_hot_x, "G V d_G, b s G V -> b s G d_G")

    def _calc_diversity_loss(
        self, x_logits: torch.Tensor, xlens: torch.Tensor
    ) -> torch.Tensor:
        x_logits_groupby = self._logits_group_by_codebook(x_logits)

        # diversity loss
        x_prob_groupby = F.softmax(x_logits_groupby, dim=-1)

        valid_mask = rearrange(
            self._lens_to_valid_mask(xlens, max_len=x_prob_groupby.size(1)),
            "b s -> b s 1 1",
        ).to(torch.float)

        x_prob_groupby_masked = x_prob_groupby * valid_mask

        x_prob_g_v = (
            reduce(x_prob_groupby_masked, "b s G V -> G V", "sum") / xlens.sum()
        )

        return reduce(torch.xlogy(x_prob_g_v, x_prob_g_v), "G V -> ", "mean")

    def forward(
        self, x: torch.Tensor, xlens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_logits = self.quantization_linear_in(x)

        # select codebooks with gumbel softmax
        selected_codebooks = self._select_codebooks(x_logits)
        selected_codebooks_concat = rearrange(
            selected_codebooks, "b s G d_G -> b s (G d_G)"
        )
        out = self.quantization_proj_out(selected_codebooks_concat)

        # calc diversity loss
        diversity_loss = self._calc_diversity_loss(x_logits, xlens)

        return out, xlens, diversity_loss
