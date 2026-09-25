import torch
from torch import nn


class SpecAugment(nn.Module):
    def __init__(
        self,
        time_mask_prob: float,
        time_mask_length: int,
        channel_mask_prob: float,
        channel_mask_length: int,
    ) -> None:
        super().__init__()
        self.time_mask_prob = time_mask_prob
        self.time_mask_length = time_mask_length
        self.channel_mask_prob = channel_mask_prob
        self.channel_mask_length = channel_mask_length

    @staticmethod
    def _mask_along_axis(
        valid_sizes: torch.Tensor,
        axis_size: int,
        mask_prob: float,
        mask_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        batch_size = valid_sizes.shape[0]
        mask = torch.zeros(batch_size, axis_size, dtype=torch.bool, device=device)
        if mask_prob <= 0 or mask_length <= 0:
            return mask

        for i in range(batch_size):
            size = int(valid_sizes[i])
            if size <= mask_length:
                continue
            num_spans = int(mask_prob * size / mask_length + torch.rand(1).item())
            for _ in range(num_spans):
                start = int(torch.randint(0, size - mask_length + 1, (1,)).item())
                mask[i, start : start + mask_length] = True
        return mask

    def forward(
        self,
        feature_lens: torch.Tensor,
        seq_len: int,
        num_channels: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        time_mask = self._mask_along_axis(
            feature_lens, seq_len, self.time_mask_prob, self.time_mask_length, device
        )
        channel_valid = torch.full_like(feature_lens, num_channels)
        channel_mask = self._mask_along_axis(
            channel_valid,
            num_channels,
            self.channel_mask_prob,
            self.channel_mask_length,
            device,
        )
        return time_mask, channel_mask
