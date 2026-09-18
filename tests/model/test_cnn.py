from dataclasses import dataclass

import pytest
import torch

from wav2vec2.model.modules.cnn import Wav2Vec2FeatureEncoder
from wav2vec2.model.modules.schemas import Wav2Vec2FeatureEncoderConfig


@pytest.fixture
def encoder() -> Wav2Vec2FeatureEncoder:
    torch.manual_seed(0)
    return Wav2Vec2FeatureEncoder(config=Wav2Vec2FeatureEncoderConfig()).eval()


@pytest.mark.parametrize(
    ("num_samples", "expected_frames"),
    [
        (400, 1),
        (719, 1),
        (720, 2),
        (16000, 49),
        (16080, 50),
        (32000, 99),
    ],
)
def test_output_length(
    encoder: Wav2Vec2FeatureEncoder, num_samples: int, expected_frames: int
) -> None:
    x = torch.randn(1, num_samples)
    xlens = torch.tensor([num_samples])

    y, ylens = encoder(x, xlens)

    assert y.shape == (1, expected_frames, 512)
    assert ylens.tolist() == [expected_frames]


@dataclass(frozen=True)
class PaddedBatch:
    x: torch.Tensor
    xlens: torch.Tensor


@pytest.fixture(
    params=[
        [9600, 16000, 2400, 4800],
        [16000, 16000],
        [400, 32000],
    ],
)
def padded_batch(request: pytest.FixtureRequest) -> PaddedBatch:
    lens: list[int] = request.param
    torch.manual_seed(0)
    x = torch.randn(len(lens), max(lens))
    for i, n in enumerate(lens):
        x[i, n:] = 0.0
    return PaddedBatch(x=x, xlens=torch.tensor(lens))


def test_padding_does_not_affect_valid_frames(
    encoder: Wav2Vec2FeatureEncoder, padded_batch: PaddedBatch
) -> None:
    y_batch, ylens = encoder(padded_batch.x, padded_batch.xlens)

    for i, n in enumerate(padded_batch.xlens.tolist()):
        y_single, _ = encoder(padded_batch.x[i : i + 1, :n], torch.tensor([n]))
        torch.testing.assert_close(y_batch[i, : ylens[i]], y_single[0])
