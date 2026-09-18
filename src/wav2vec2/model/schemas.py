from dataclasses import dataclass

from .modules.schemas import (
    Wav2Vec2AdapterConfig,
    Wav2Vec2FeatureEncoderConfig,
    Wav2Vec2TransformerConfig,
)


@dataclass(frozen=True)
class Wav2Vec2Config:
    feature_encoder_config: Wav2Vec2FeatureEncoderConfig
    adapter_config: Wav2Vec2AdapterConfig
    transformer_config: Wav2Vec2TransformerConfig

    feature_encoder_dropout_prob: float = 0.1
