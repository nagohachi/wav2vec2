from dataclasses import dataclass


@dataclass(frozen=True)
class Wav2Vec2FeatureEncoderConfig:
    in_channels: int = 1
    out_channels: tuple[int, ...] = (512, 512, 512, 512, 512, 512, 512)
    kernel_sizes: tuple[int, ...] = (10, 3, 3, 3, 3, 2, 2)
    strides: tuple[int, ...] = (5, 2, 2, 2, 2, 2, 2)
