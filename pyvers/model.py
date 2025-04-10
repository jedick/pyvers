import os
import torch
import pytorch_lightning as pl
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
#import numpy as np

class PyversClassifier(pl.LightningModule):
    def __init__(
        self, 
        model_name: str,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        id2label: dict = {0:"SUPPORT", 1:"NEI", 2:"REFUTE"},
        label2id: dict = {"SUPPORT":0, "NEI":1, "REFUTE":2},
        tensorboard_logdir: str = "tb_logs",
        **kwargs,
    ):
        super().__init__()

        # Save hyperparameters (lets us use self.hparams)
        self.save_hyperparameters()
        
        # Model
        num_classes = len(id2label)
        config = AutoConfig.from_pretrained(model_name, num_labels=num_classes)
        config.hidden_dropout_prob = hidden_dropout_prob
        config.attention_probs_dropout_prob = attention_probs_dropout_prob
        config.id2label = id2label
        config.label2id = label2id
        # HuggingFace pretrained models are in eval mode by default (Dropout modules are deactivated)
        #   https://huggingface.co/docs/transformers/en/main_classes/model#transformers.PreTrainedModel.from_pretrained
        # We have to put the model in train mode to make dropout work
        #   https://github.com/Lightning-AI/pytorch-lightning/issues/20105
        #   https://github.com/Lightning-AI/pytorch-lightning/issues/20646
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config).train()

        # Metrics
        self.train_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        # Keep F1 for test split separate in order to log individual classes
        self.test_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_classes, average=None)
        # For remaining test metrics, use MetricCollection
        self.test_metrics = torchmetrics.MetricCollection(
            {
                "Accuracy": torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes),
                "F1 Micro": torchmetrics.classification.F1Score(task="multiclass", num_classes=num_classes, average="micro"),
                "F1 Macro": torchmetrics.classification.F1Score(task="multiclass", num_classes=num_classes, average="macro"),
                "AUROC Macro": torchmetrics.classification.AUROC(task="multiclass", num_classes=num_classes, average="macro"),
                "AUROC Weighted": torchmetrics.classification.AUROC(task="multiclass", num_classes=num_classes, average="weighted"),
            },
        )

        # Log train and val metrics to different directories to plot them on one graph in TensorBoard
        self.train_writer = SummaryWriter(os.path.join(self.hparams.tensorboard_logdir, "train"))
        self.val_writer = SummaryWriter(os.path.join(self.hparams.tensorboard_logdir, "val"))

        self.train_losses = []
        self.val_losses = []

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

    @staticmethod
    def _percent(x):
        ## TODO: How to log rounded values without extraneous decimals?
        ## float16 avoids long decimals but is not precise enough to be of use.
        #x_float16 = np.float16(x.detach().cpu().numpy())
        #return round(100*x_float16, 2)
        return torch.round(100*x, decimals=2)

    def _log_metrics(self, train_or_val, accuracy, losses, writer):
        # Compute and log accuracy
        train_or_val_accuracy = self._percent(accuracy.compute())
        self.log(f"{train_or_val}_accuracy", train_or_val_accuracy)
        # Write accuracy to TensorBoard logger
        writer.add_scalar("accuracy", train_or_val_accuracy, self.current_epoch)
        accuracy.reset()
        # Compute and log loss
        train_or_val_loss = torch.stack([x for x in losses]).mean()
        self.log(f"{train_or_val}_loss", train_or_val_loss, prog_bar=False)
        # Write loss to TensorBoard logger
        writer.add_scalar("loss", train_or_val_loss, self.current_epoch)
        losses.clear()

    def training_step(self, batch, batch_idx):
        # Run the forward pass
        outputs = self(**batch)
        # Calculate the loss
        loss = outputs.loss
        # Display the loss at this step and accumulate it for batch average
        self.log("train_loss", loss, prog_bar=True, logger=False)
        self.train_losses.append(loss)
        # Update accuracy metric
        y = batch.get("labels")
        self.train_accuracy.update(outputs.logits, y)
        return loss

    def on_train_epoch_end(self):
        self._log_metrics("train", self.train_accuracy, self.train_losses, self.train_writer)

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("val_loss", loss, prog_bar=True, logger=False)
        self.val_losses.append(loss)
        y = batch.get("labels")
        self.val_accuracy.update(outputs.logits, y)

    def on_validation_epoch_end(self):
        self._log_metrics("val", self.val_accuracy, self.val_losses, self.val_writer)

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        y = batch.get("labels")
        self.test_f1.update(outputs.logits, y)
        self.test_metrics.update(outputs.logits, y)

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())
        # Log F1 score for each class
        test_f1 = self.test_f1.compute()
        num_classes = len(self.hparams.id2label)
        for id in range(num_classes):
            label = self.hparams.id2label[id]
            self.log(f"F1_{label}", test_f1[id])
        self.test_f1.reset()
        self.test_metrics.reset()

    def predict_step(self, batch, batch_idx):
        outputs = self(**batch)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_ids = torch.argmax(probabilities, dim=1).tolist()
        predicted_labels = [self.hparams.id2label[id] for id in predicted_ids]
        return predicted_labels

