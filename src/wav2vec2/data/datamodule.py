from pathlib import Path
from typing import Literal

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from .dataset import LibriSpeechDataset


class LibriSpeechDataModule(LightningDataModule):
    def __init__(
        self,
        data_root: Path,
        train_split: Literal["train-clean-100", "train-960"],
        batch_size: int = 8,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.train_split: Literal["train-clean-100", "train-960"] = train_split
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = LibriSpeechDataset(self.data_root, self.train_split)
        self.val_dataset = LibriSpeechDataset(self.data_root, "dev-clean")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=LibriSpeechDataset.collate_fn,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=LibriSpeechDataset.collate_fn,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )
