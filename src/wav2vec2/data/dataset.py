import json
import string
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import soundfile
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

TRAIN_960_SUBSPLITS = ["train-clean-100", "train-clean-360", "train-other-500"]


@dataclass(frozen=True)
class LibriSpeechSample:
    utterance_id: str
    waveform: torch.Tensor  # (num_samples, )
    transcript: str


@dataclass
class LibriSpeechBatch:
    utterance_ids: list[str]
    waveforms: torch.Tensor  # (batch_size, max_num_samples)
    waveform_lens: torch.Tensor  # (batch_size, )
    transcripts: list[str]
    targets: torch.Tensor  # (batch_size, max_target_len)
    target_lens: torch.Tensor  # (batch_size, )


class LibriSpeechDataset(Dataset):
    LIBRISPEECH_SAMPLE_RATE = 16_000

    CTC_BLANK_IDX = 0
    VOCAB = "_ '" + string.ascii_uppercase  # _ is the CTC blank

    def __init__(
        self,
        data_root: Path,
        split: Literal["train-clean-100", "dev-clean", "train-960"],
    ) -> None:
        sub_splits = TRAIN_960_SUBSPLITS if split == "train-960" else [split]

        for sub_split in sub_splits:
            if not (data_root / "LibriSpeech" / sub_split).exists():
                raise FileNotFoundError(
                    "Dataset does not exist. Please download Librispeech first"
                )

        self._utterance_ids: list[str] = []
        self._flac_paths: list[Path] = []
        self._transcripts: list[str] = []

        for sub_split in sub_splits:
            split_root = data_root / "LibriSpeech" / sub_split
            for trans_path in sorted(split_root.rglob("*.trans.txt")):
                chapter_dir = trans_path.parent
                for line in trans_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    utterance_id, transcript = line.split(" ", 1)
                    self._utterance_ids.append(utterance_id)
                    self._flac_paths.append(chapter_dir / f"{utterance_id}.flac")
                    self._transcripts.append(transcript)

        cache_path = data_root / "LibriSpeech" / f"frame_lengths.{split}.json"
        self.frame_lengths = self._load_or_build_frame_lengths(cache_path)

    def _load_or_build_frame_lengths(self, cache_path: Path) -> list[int]:
        if cache_path.exists():
            cached = json.loads(cache_path.read_text())
            if all(uid in cached for uid in self._utterance_ids):
                return [cached[uid] for uid in self._utterance_ids]

        with ThreadPoolExecutor() as executor:
            frame_lengths = list(
                executor.map(lambda p: soundfile.info(p).frames, self._flac_paths)
            )

        cache_path.write_text(json.dumps(dict(zip(self._utterance_ids, frame_lengths))))
        return frame_lengths

    def __len__(self) -> int:
        return len(self._flac_paths)

    def __getitem__(self, index: int) -> LibriSpeechSample:
        audio, sample_rate = soundfile.read(self._flac_paths[index], dtype="float32")
        assert sample_rate == self.LIBRISPEECH_SAMPLE_RATE, (
            f"expected {self.LIBRISPEECH_SAMPLE_RATE} Hz, got {sample_rate} Hz"
        )

        return LibriSpeechSample(
            utterance_id=self._utterance_ids[index],
            waveform=torch.from_numpy(audio),
            transcript=self._transcripts[index],
        )

    @classmethod
    def encode_transcript(cls, transcript: str) -> torch.Tensor:
        return torch.tensor([cls.VOCAB.index(char) for char in transcript])

    @classmethod
    def collate_fn(cls, samples: list[LibriSpeechSample]) -> LibriSpeechBatch:
        targets = [cls.encode_transcript(s.transcript) for s in samples]

        return LibriSpeechBatch(
            utterance_ids=[s.utterance_id for s in samples],
            waveforms=pad_sequence([s.waveform for s in samples], batch_first=True),
            waveform_lens=torch.tensor([s.waveform.shape[0] for s in samples]),
            transcripts=[s.transcript for s in samples],
            targets=pad_sequence(
                targets, batch_first=True, padding_value=cls.CTC_BLANK_IDX
            ),
            target_lens=torch.tensor([t.shape[0] for t in targets]),
        )
