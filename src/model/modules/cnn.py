import torch
from einops import rearrange
from torch import nn

from .schemas import Wav2Vec2FeatureEncoderConfig


class ChannelLayerNorm(nn.Module):
    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (batch_size, channels, seq_lens)
        """
        x = rearrange(x, "b c s -> b s c")
        x = self.norm(x)
        return rearrange(x, "b s c -> b c s")


class Wav2Vec2FeatureEncoder(nn.Module):
    def __init__(self, config: Wav2Vec2FeatureEncoderConfig) -> None:
        super().__init__()

        self.config = config

        _layers = []
        _in_channels = [config.in_channels, *config.out_channels[:-1]]

        for in_channels, out_channels, kernel_size, stride in zip(
            _in_channels, config.out_channels, config.kernel_sizes, config.strides
        ):
            _layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel_size, stride),
                    ChannelLayerNorm(out_channels),
                    nn.GELU(),
                ]
            )

        self.cnn = nn.Sequential(*_layers)

    def _calc_output_len_for_conv1d(
        self,
        input_lens: torch.Tensor,
        kernel_size: int,
        stride: int,
        padding: int = 0,  # set if you have padding
        dilation: int = 1,  # set if you have dilation
    ) -> torch.Tensor:
        return (
            input_lens + 2 * padding - dilation * (kernel_size - 1) - 1
        ) // stride + 1

    def _calc_output_len(self, input_lens: torch.Tensor) -> torch.Tensor:
        for kernel_size, stride in zip(self.config.kernel_sizes, self.config.strides):
            input_lens = self._calc_output_len_for_conv1d(
                input_lens, kernel_size, stride
            )

        return input_lens

    def forward(
        self, x: torch.Tensor, xlens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): input tensor of (batch_size, seq_len).
            xlens (torch.Tensor): tensor describing length of (batch_size, )

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (out_x, out_x_lens)
            out_x: (batch_size, seq_len', hid_dim)
        """
        x = rearrange(x, "b s -> b 1 s")
        x = self.cnn(x)
        return rearrange(x, "b h s -> b s h"), self._calc_output_len(xlens)
