import os
import torch
import pytorch_lightning as pl
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
import numpy as np

class PyversClassifier(pl.LightningModule):
    # Use label IDs from MultiVerS
    def __init__(
        self, 
        model_name: str,
        ## MultiVerS
        #id2label: dict = {0:"REFUTE", 1:"NEI", 2:"SUPPORT"},
        # https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
        # label_names = ["entailment", "neutral", "contradiction"]
        id2label: dict = {0:"SUPPORT", 1:"NEI", 2:"REFUTE"},
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()
        
        num_classes = len(id2label)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
        self.test_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_classes, average=None)

    def forward(self, **inputs):
        return self.model(**inputs)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        y = batch.get("labels")
        self.test_accuracy.update(outputs.logits, y)
        self.test_f1.update(outputs.logits, y)

    def on_test_epoch_end(self):
        test_accuracy = self.test_accuracy.compute()
        test_f1 = self.test_f1.compute()
        self.log("Accuracy", round(100*test_accuracy.detach().cpu().numpy()))
        # Log F1 score for each class
        num_classes = len(self.hparams.id2label)
        for id in range(num_classes):
            label = self.hparams.id2label[id]
            self.log(f"F1_{label}", round(100*test_f1[id].detach().cpu().numpy()))
        self.test_accuracy.reset()
        self.test_f1.reset()

    def predict_step(self, batch, batch_idx):
        outputs = self(**batch)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_ids = torch.argmax(predictions, dim=1).tolist()
        predicted_labels = [self.hparams.id2label[id] for id in predicted_ids]
        return predicted_labels

class MetricLogger(pl.Callback):

    def __init__(self):
        # Log train and val losses to different directories to plot them on one graph in TensorBoard
        logdir = "tb_logs"
        self.train_writer = SummaryWriter(os.path.join(logdir, "train"))
        self.val_writer = SummaryWriter(os.path.join(logdir, "val"))

    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.logged_metrics["train_loss"]
        self.train_writer.add_scalar("loss", train_loss, trainer.current_epoch)

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.logged_metrics["val_loss"]
        self.val_writer.add_scalar("loss", val_loss, trainer.current_epoch)

