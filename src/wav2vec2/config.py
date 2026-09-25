from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import dacite
import yaml

from .model.schemas import Wav2Vec2Config


@dataclass
class DataConfig:
    data_root: Path
    train_split: Literal["train-clean-100", "train-960"] = "train-clean-100"
    batch_seconds: float = 200.0
    num_workers: int = 4


@dataclass
class FinetuneConfig:
    learning_rate: float = 3e-5
    adam_betas: tuple[float, float] = (0.9, 0.98)
    adam_eps: float = 1e-8
    weight_decay: float = 0.0
    max_steps: int = 80000
    warmup_ratio: float = 0.1
    hold_ratio: float = 0.4
    freeze_feature_encoder: bool = True
    freeze_transformer_steps: int = 10000
    time_mask_prob: float = 0.65
    time_mask_length: int = 10
    channel_mask_prob: float = 0.5
    channel_mask_length: int = 64


@dataclass
class TrainConfig:
    data: DataConfig
    model: Wav2Vec2Config = field(default_factory=Wav2Vec2Config)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    trainer: dict[str, Any] = field(default_factory=dict)


def load_config(path: Path) -> TrainConfig:
    raw = yaml.safe_load(path.read_text()) or {}
    return dacite.from_dict(
        TrainConfig,
        raw,
        config=dacite.Config(type_hooks={Path: Path}, cast=[tuple], strict=True),
    )
