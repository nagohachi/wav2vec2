import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import nn

from ..data.dataset import LibriSpeechBatch, LibriSpeechDataset
from ..model.schemas import Wav2Vec2Config
from ..model.wav2vec2 import Wav2Vec2ForCTC


class Wav2Vec2CTCModule(LightningModule):
    def __init__(
        self,
        config: Wav2Vec2Config,
        lr: float = 3e-4,
        vocab_size: int = len(LibriSpeechDataset.VOCAB),
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.model = Wav2Vec2ForCTC(config, vocab_size)
        self.ctc_loss = nn.CTCLoss(
            blank=LibriSpeechDataset.CTC_BLANK_IDX, zero_infinity=True
        )
        self.lr = lr

    def _step(self, batch: LibriSpeechBatch) -> torch.Tensor:
        logits, output_lens = self.model(batch.waveforms, batch.waveform_lens)
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
        return self.ctc_loss(log_probs, batch.targets, output_lens, batch.target_lens)

    def training_step(self, batch: LibriSpeechBatch, batch_idx: int) -> torch.Tensor:
        loss = self._step(batch)
        self.log("train_loss", loss, prog_bar=True, batch_size=len(batch.transcripts))
        return loss

    def validation_step(self, batch: LibriSpeechBatch, batch_idx: int) -> torch.Tensor:
        loss = self._step(batch)
        self.log("val_loss", loss, prog_bar=True, batch_size=len(batch.transcripts))
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
