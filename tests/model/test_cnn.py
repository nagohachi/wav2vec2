import torch

from wav2vec2.model.modules.cnn import Wav2Vec2FeatureEncoder
from wav2vec2.model.modules.schemas import Wav2Vec2FeatureEncoderConfig


def test_temporal_reduction() -> None:
    config = Wav2Vec2FeatureEncoderConfig()
    cnn = Wav2Vec2FeatureEncoder(config=config)

    x = torch.randn(size=(4, 16000))
    xlens = torch.tensor([9600, 16000, 2400, 4800], dtype=torch.long)

    y = cnn(x, xlens)

    assert y[0].shape == (4, 49, 512)
    assert y[1].tolist() == [29, 49, 7, 14]
