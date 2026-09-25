import argparse
from pathlib import Path

from lightning import Trainer

from .config import TrainConfig, load_config
from .data.datamodule import LibriSpeechDataModule
from .framework.ctc_framework import Wav2Vec2CTCModule


def sl(cfg: TrainConfig) -> None:
    datamodule = LibriSpeechDataModule(
        data_root=cfg.data.data_root,
        train_split=cfg.data.train_split,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )
    module = Wav2Vec2CTCModule(
        model_config=cfg.model, finetune_config=cfg.finetune
    )
    Trainer(**cfg.trainer).fit(module, datamodule)


def ssl(_cfg: TrainConfig) -> None:
    raise NotImplementedError("SSL training is not implemented yet")


def main() -> None:
    parser = argparse.ArgumentParser(prog="wav2vec2")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("sl", "ssl"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("config", type=Path)

    args = parser.parse_args()
    cfg = load_config(args.config)
    {"sl": sl, "ssl": ssl}[args.command](cfg)
