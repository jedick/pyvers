import os
import torch
import pytorch_lightning as pl
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForSequenceClassification

class BERTClassifier(pl.LightningModule):
    # Use label IDs from MultiVerS
    def __init__(self, id2label = {0:"REFUTE", 1:"NEI", 2:"SUPPORT"}):
        super().__init__()
        model_path = "bert-base-uncased"
        self.id2label = id2label
        self.num_classes = len(id2label)

        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=self.num_classes)
        self.test_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=self.num_classes, average=None)

    def forward(self, batch):
        input_ids = batch["input_ids"]
        token_type_ids = batch["token_type_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        return self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-5)

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        outputs = self(batch)
        y = batch.get("labels")
        self.test_accuracy.update(outputs.logits, y)
        self.test_f1.update(outputs.logits, y)
        #loss = outputs.loss
        #self.log("test_loss", loss, prog_bar=True)
        #return loss

    def on_test_epoch_end(self):
        test_accuracy = self.test_accuracy.compute()
        test_f1 = self.test_f1.compute()
        self.log("Accuracy", round(100*test_accuracy.item()))
        # Log F1 score for each class
        for id in range(self.num_classes):
            label = self.id2label[id]
            self.log(f"F1_{label}", round(100*test_f1[id].item()))
        self.test_accuracy.reset()
        self.test_f1.reset()

    def predict_step(self, batch, batch_idx):
        outputs = self(batch)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_ids = torch.argmax(predictions, dim=1).tolist()
        predicted_labels = [self.id2label[id] for id in predicted_ids]
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

