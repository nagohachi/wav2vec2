from dataclasses import dataclass


@dataclass(frozen=True)
class Wav2Vec2FeatureEncoderConfig:
    in_channels: int = 1
    out_channels: tuple[int, ...] = (512, 512, 512, 512, 512, 512, 512)
    kernel_sizes: tuple[int, ...] = (10, 3, 3, 3, 3, 2, 2)
    strides: tuple[int, ...] = (5, 2, 2, 2, 2, 2, 2)


@dataclass(frozen=True)
class Wav2Vec2AdapterConfig:
    in_dim: int = 512
    out_dim: int = 768


@dataclass(frozen=True)
class Wav2Vec2TransformerConfig:
    hidden_dim: int = 768
    ffn_hidden_dim: int = 3072
    num_layers: int = 12
    num_attention_heads: int = 8


@dataclass(frozen=True)
class Wav2Vec2QuantizationConfig:
    num_code_group: int = 2  # G
    codebook_size_per_group: int = 320  # V
    concat_codebook_dim: int = 256  # d, codebook hidden dim = d/G

    quantization_in_dim: int = 512  # must be feature enc's output dim
    quantization_out_dim: int = 768  # must be transformer's hidden dim

    def __post_init__(self) -> None:
        if self.concat_codebook_dim % self.num_code_group != 0:
            raise ValueError("concat_codebook_dim must be divisible by num_code_group")
