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
    batch_size: int = 8
    num_workers: int = 4


@dataclass
class OptimConfig:
    lr: float = 3e-4


@dataclass
class TrainConfig:
    data: DataConfig
    model: Wav2Vec2Config = field(default_factory=Wav2Vec2Config)
    optim: OptimConfig = field(default_factory=OptimConfig)
    trainer: dict[str, Any] = field(default_factory=dict)


def load_config(path: Path) -> TrainConfig:
    raw = yaml.safe_load(path.read_text()) or {}
    return dacite.from_dict(
        TrainConfig,
        raw,
        config=dacite.Config(type_hooks={Path: Path}, cast=[tuple], strict=True),
    )
