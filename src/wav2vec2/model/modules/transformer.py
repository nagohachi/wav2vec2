import torch
from einops import rearrange
from torch import nn

from .schemas import Wav2Vec2TransformerConfig


class ConvPositionalEmbedding(nn.Module):
    def __init__(
        self, hidden_dim: int, kernel_size: int = 128, groups: int = 16
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2, groups=groups
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch_size, seq_len, hidden_dim)"""
        h = self.conv(rearrange(x, "b t d -> b d t"))
        if self.kernel_size % 2 == 0:
            # if kernel_size % 2 == 0, one frame added:
            #     https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv1d.html
            # not compatible with x = x + pos_emb(x)
            h = h[..., :-1]
        return rearrange(self.activation(h), "b d t -> b t d")


class Wav2Vec2Transformer(nn.Module):
    def __init__(self, config: Wav2Vec2TransformerConfig) -> None:
        super().__init__()
        self.conv_pos_emb = ConvPositionalEmbedding(hidden_dim=config.hidden_dim)
        self.layernorm_after_pos_emb = nn.LayerNorm(config.hidden_dim)

        _encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_attention_heads,
            dim_feedforward=config.ffn_hidden_dim,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=_encoder_layer, num_layers=config.num_layers
        )

    @staticmethod
    def _lens_to_mask(xlens: torch.Tensor, max_len: int) -> torch.Tensor:
        """converts xlens of (batch_size, ) to pytorch-style src-key-padding mask."""
        positions = torch.arange(max_len, device=xlens.device)
        return positions[None, :] >= xlens[:, None]

    def forward(
        self, x: torch.Tensor, xlens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): input tensor of (batch_size, seq_len, hidden_dim).
            xlens (torch.Tensor): tensor describing length of (batch_size, )

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (out_x, out_x_lens)
            out_x: (batch_size, seq_len, hidden_dim)
        """
        mask = self._lens_to_mask(xlens, x.size(1))

        mask_3d = rearrange(mask, "b s -> b s 1")

        # positional embedding
        # 0.0 where pad (so that conv does not take effect)
        x = x.masked_fill(mask_3d, 0.0)
        x = x + self.conv_pos_emb(x)
        x = self.layernorm_after_pos_emb(x)

        # transformer
        x = self.transformer(x, src_key_padding_mask=mask)
        return x, xlens
