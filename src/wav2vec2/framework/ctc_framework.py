import torch
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn

from ..config import FinetuneConfig
from ..data.dataset import LibriSpeechBatch, LibriSpeechDataset
from ..model.modules.spec_augment import SpecAugment
from ..model.schemas import Wav2Vec2Config
from ..model.wav2vec2 import Wav2Vec2ForCTC


class Wav2Vec2CTCModule(LightningModule):
    def __init__(
        self,
        model_config: Wav2Vec2Config,
        finetune_config: FinetuneConfig,
        vocab_size: int = len(LibriSpeechDataset.VOCAB),
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model_config"])
        self.finetune_config = finetune_config
        spec_augment = SpecAugment(
            finetune_config.time_mask_prob,
            finetune_config.time_mask_length,
            finetune_config.channel_mask_prob,
            finetune_config.channel_mask_length,
        )
        self.model = Wav2Vec2ForCTC(model_config, vocab_size, spec_augment=spec_augment)
        self.ctc_loss = nn.CTCLoss(
            blank=LibriSpeechDataset.CTC_BLANK_IDX, zero_infinity=True
        )
        if finetune_config.freeze_feature_encoder:
            self._freeze_feature_encoder()

    def _freeze_feature_encoder(self) -> None:
        for param in self.model.wav2vec2.feature_encoder.parameters():
            param.requires_grad_(False)

    def _set_transformer_trainable(self, trainable: bool) -> None:
        wav2vec2 = self.model.wav2vec2
        wav2vec2.mask_emb.requires_grad_(trainable)
        for module in (wav2vec2.adapter, wav2vec2.transformer):
            for param in module.parameters():
                param.requires_grad_(trainable)

    def on_train_start(self) -> None:
        thawed = self.global_step >= self.finetune_config.freeze_transformer_steps
        self._set_transformer_trainable(thawed)

    def on_train_batch_start(self, batch: LibriSpeechBatch, batch_idx: int) -> None:
        if self.global_step == self.finetune_config.freeze_transformer_steps:
            self._set_transformer_trainable(True)

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

    def _lr_scale(self, step: int) -> float:
        finetune = self.finetune_config
        warmup = int(finetune.max_steps * finetune.warmup_ratio)
        hold = int(finetune.max_steps * finetune.hold_ratio)
        if step < warmup:
            return step / max(1, warmup)
        if step < warmup + hold:
            return 1.0
        decay = finetune.max_steps - warmup - hold
        return max(0.0, 1.0 - (step - warmup - hold) / max(1, decay))

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.finetune_config.learning_rate,
            betas=self.finetune_config.adam_betas,
            eps=self.finetune_config.adam_eps,
            weight_decay=self.finetune_config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self._lr_scale)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
