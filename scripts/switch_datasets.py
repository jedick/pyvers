import torch
import pytorch_lightning as pl
from pyvers.data import FileDataModule
from pyvers.model import PyversClassifier
from pytorch_lightning.loggers import CSVLogger

# Running float32 matrix multiplications in lower precision may significantly increase performance
# https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("medium")

# Define datasets to train on
first = "scifact"
second = "citint"
csv_logger = CSVLogger("experiments", name="switch_datasets")

# Setup model and trainer
model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
model = PyversClassifier(model_name, tensorboard_logdir=f"experiments/switch_datasets")
# Train on first dataset for 10 epochs
dm = FileDataModule(f"data/{first}", model_name, batch_size=8)
trainer = pl.Trainer(
    enable_checkpointing=False,
    logger=csv_logger,
    num_sanity_val_steps=0,
    max_epochs=10,
)
trainer.fit(model, datamodule=dm)
# Save checkpoint
trainer.save_checkpoint(f"~/.checkpoints/pyvers/{first}.ckpt")

# Train on second dataset for 10 epochs
model = PyversClassifier(model_name, tensorboard_logdir=f"experiments/switch_datasets")
dm = FileDataModule(f"data/{second}", model_name, batch_size=8)
# We set max_epochs=20 because the trainer starts from the last training step in the checkpoint
trainer = pl.Trainer(
    enable_checkpointing=False,
    logger=csv_logger,
    num_sanity_val_steps=0,
    max_epochs=20,
)
trainer.fit(model, datamodule=dm, ckpt_path=f"~/.checkpoints/pyvers/{first}.ckpt")
# Save checkpoint
trainer.save_checkpoint(f"~/.checkpoints/pyvers/{first}_{second}.ckpt")

