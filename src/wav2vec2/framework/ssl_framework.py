import torch
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn

from wav2vec2.model.modules.quantization import Wav2Vec2Quantization
from wav2vec2.model.modules.schemas import Wav2Vec2QuantizationConfig

from ..config import FinetuneConfig
from ..data.dataset import LibriSpeechBatch, LibriSpeechDataset
from ..data.sampler import DynamicBatchSampler
from ..model.modules.spec_augment import SpecAugment
from ..model.schemas import Wav2Vec2Config
from ..model.wav2vec2 import Wav2Vec2


class Wav2Vec2SSLModule(LightningModule):
    def __init__(self, model_config: Wav2Vec2Config, quantization_config: Wav2Vec2QuantizationConfig) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model_config"])
        self.model = Wav2Vec2(model_config)
        self.quantization_module = Wav2Vec2Quantization(quantization_config)

    def forward(self, x: torch.Tensor, xlens: torch.Tensor):
        
