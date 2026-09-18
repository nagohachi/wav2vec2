import torch
from einops import rearrange
from torch import nn

from .modules.adapter import Wav2Vec2Adapter
from .modules.cnn import Wav2Vec2FeatureEncoder
from .modules.transformer import Wav2Vec2Transformer
from .schemas import Wav2Vec2Config


class Wav2Vec2(nn.Module):
    def __init__(self, config: Wav2Vec2Config) -> None:
        super().__init__()
        self.feature_encoder = Wav2Vec2FeatureEncoder(config.feature_encoder_config)
        self.dropout_after_feature_enc = nn.Dropout(
            p=config.feature_encoder_dropout_prob
        )
        self.adapter = Wav2Vec2Adapter(config.adapter_config)
        self.transformer = Wav2Vec2Transformer(config.transformer_config)

        # masking for feature encoder, like BERT's [MASK]
        self.mask_emb = nn.Parameter(
            torch.empty(config.feature_encoder_config.out_channels[-1]).uniform_()
        )

    def forward(
        self,
        x: torch.Tensor,
        xlens: torch.Tensor,
        time_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """forward path of wav2vec2 encoder. return feature encoder output for self-supervised learning.

        Args:
            x (torch.Tensor): input tensor of (batch_size, seq_len)
            xlens (torch.Tensor): lengths of input tensor of (batch_size, )
            time_mask (torch.Tensor): mask for feature encoder output of (batch_size, seq_len').
                None during fine-tuning or inference

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            transformer output, length, feature encoder output for the quantizer (batch_size, seq_len', cnn_hidden_size)
        """
        feature_encoder_output, feature_encoder_output_lens = self.feature_encoder(
            x, xlens
        )

        feature_encoder_output_maybe_masked = self.dropout_after_feature_enc(
            feature_encoder_output
        )
        if time_mask is not None:
            time_mask_3d = rearrange(time_mask, "b s -> b s 1")
            feature_encoder_output_maybe_masked = torch.where(
                time_mask_3d, self.mask_emb, feature_encoder_output_maybe_masked
            )

        adapter_output, adapter_output_lens = self.adapter(
            feature_encoder_output_maybe_masked, feature_encoder_output_lens
        )
        transformer_output, transformer_output_lens = self.transformer(
            adapter_output, adapter_output_lens
        )

        return transformer_output, transformer_output_lens, feature_encoder_output
