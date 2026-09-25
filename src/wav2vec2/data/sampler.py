from collections.abc import Iterator

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


def _world_size_and_rank() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    return 1, 0


class DynamicBatchSampler(Sampler[list[int]]):
    """Groups indices so each batch's padded footprint (batch size x longest
    frame length) stays within ``max_frames``; shorter utterances pack into
    larger batches.

    Batches are fixed length-buckets whose order is reshuffled every epoch;
    the owner must call ``set_epoch`` before each epoch to advance the shuffle.
    Under distributed training every rank takes an equal, disjoint slice of
    batches and the remainder is dropped so all ranks step in lockstep.
    """

    def __init__(
        self,
        frame_lengths: list[int],
        max_frames: int,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")

        self.frame_lengths = frame_lengths
        self.max_frames = max_frames
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.batches = self._bucket_by_length()

    def _bucket_by_length(self) -> list[list[int]]:
        order = sorted(range(len(self.frame_lengths)), key=self.frame_lengths.__getitem__)

        batches: list[list[int]] = []
        batch: list[int] = []
        longest = 0
        for index in order:
            length = self.frame_lengths[index]
            longest_if_added = max(longest, length)
            if batch and (len(batch) + 1) * longest_if_added > self.max_frames:
                batches.append(batch)
                batch, longest_if_added = [], length
            batch.append(index)
            longest = longest_if_added
        if batch:
            batches.append(batch)
        return batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _epoch_order(self) -> list[list[int]]:
        if not self.shuffle:
            return self.batches
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.batches), generator=generator).tolist()
        return [self.batches[i] for i in order]

    def __iter__(self) -> Iterator[list[int]]:
        batches = self._epoch_order()
        world_size, rank = _world_size_and_rank()
        usable = len(batches) // world_size * world_size
        return iter(batches[rank:usable:world_size])

    def __len__(self) -> int:
        world_size, _ = _world_size_and_rank()
        return len(self.batches) // world_size
