import torch
import pytorch_lightning as pl
from pyvers.data import FileDataModule
from pyvers.model import PyversClassifier
from pytorch_lightning.loggers import CSVLogger

# Running float32 matrix multiplications in lower precision may significantly increase performance
# https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("medium")

csv_logger = CSVLogger("experiments", name="shuffle_datasets")

# Shuffle training data from two datasets
directory = ["data/scifact", "data/citint"]
dm = FileDataModule("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", directory, batch_size=8)
model = PyversClassifier(dm.model_name, tensorboard_logdir="experiments/shuffle_datasets")
trainer = pl.Trainer(
    enable_checkpointing=False,
    logger=csv_logger, 
    num_sanity_val_steps=0,
    max_epochs=10,
)
trainer.fit(model, datamodule=dm)

# Save checkpoint
trainer.save_checkpoint(f"~/.checkpoints/pyvers/shuffle_datasets.ckpt")

