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
        id2label: dict = {0:"SUPPORT", 1:"NEI", 2:"REFUTE"},
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        **kwargs,
    ):
        super().__init__()

        # Save hyperparameters (self.hparams)
        self.save_hyperparameters()
        
        # Model
        num_classes = len(id2label)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)

        # Metrics
        self.train_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_classes, average=None)
        self.test_f1_micro = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_classes, average="micro")
        self.test_f1_macro = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_classes, average="macro")

        # Log train and val metrics to different directories to plot them on one graph in TensorBoard
        logdir = "tb_logs"
        self.train_writer = SummaryWriter(os.path.join(logdir, "train"))
        self.val_writer = SummaryWriter(os.path.join(logdir, "val"))

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
        # Compute and log accuracy
        train_accuracy = torch.round(100 * self.train_accuracy.compute(), decimals=0)
        self.log("train_accuracy", train_accuracy)
        # Write accuracy to TensorBoard logger
        self.train_writer.add_scalar("accuracy", train_accuracy, self.current_epoch)
        self.train_accuracy.reset()
        # Compute and log loss
        train_loss = torch.stack([x for x in self.train_losses]).mean()
        self.log("train_loss", train_loss, prog_bar=False)
        # Write loss to TensorBoard logger
        self.train_writer.add_scalar("loss", train_loss, self.current_epoch)
        self.train_losses.clear()

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("val_loss", loss, prog_bar=True, logger=False)
        self.val_losses.append(loss)
        y = batch.get("labels")
        self.val_accuracy.update(outputs.logits, y)

    def on_validation_epoch_end(self):
        # Compute and log accuracy
        val_accuracy = torch.round(100 * self.val_accuracy.compute(), decimals=0)
        self.log("val_accuracy", val_accuracy)
        self.val_writer.add_scalar("accuracy", val_accuracy, self.current_epoch)
        self.val_accuracy.reset()
        val_loss = torch.stack([x for x in self.val_losses]).mean()
        self.log("val_loss", val_loss, prog_bar=False)
        self.val_writer.add_scalar("loss", val_loss, self.current_epoch)
        self.val_losses.clear()

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        y = batch.get("labels")
        self.test_accuracy.update(outputs.logits, y)
        self.test_f1.update(outputs.logits, y)
        self.test_f1_micro.update(outputs.logits, y)
        self.test_f1_macro.update(outputs.logits, y)

    def on_test_epoch_end(self):
        test_accuracy = self.test_accuracy.compute()
        self.log("Accuracy", torch.round(100*test_accuracy, decimals=0))
        test_f1_micro = self.test_f1_micro.compute()
        self.log("F1 Micro", torch.round(100*test_f1_micro, decimals=0))
        test_f1_macro = self.test_f1_macro.compute()
        self.log("F1 Macro", torch.round(100*test_f1_macro, decimals=0))
        # Log F1 score for each class
        test_f1 = self.test_f1.compute()
        num_classes = len(self.hparams.id2label)
        for id in range(num_classes):
            label = self.hparams.id2label[id]
            self.log(f"F1_{label}", torch.round(100*test_f1[id], decimals=0))
        self.test_accuracy.reset()
        self.test_f1.reset()

    def predict_step(self, batch, batch_idx):
        outputs = self(**batch)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_ids = torch.argmax(predictions, dim=1).tolist()
        predicted_labels = [self.hparams.id2label[id] for id in predicted_ids]
        return predicted_labels

