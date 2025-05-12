import torch
import pytorch_lightning as pl
from pyvers.data import FileDataModule
from pyvers.model import PyversClassifier

### Calculate metrics on validation and test sets

# Read checkpoint
checkpoint = "shuffle_datasets"
model = PyversClassifier.load_from_checkpoint(
    checkpoint_path=f"/home/pyvers/checkpoints/{checkpoint}.ckpt"
)
model_name = model.hparams.model_name

# Setup dataset for testing
testdata = "scifact"
trainer = pl.Trainer()

# Make predictions on validation set
dm = FileDataModule(f"data/{testdata}", model_name, batch_size=8, use_val_for_test=True)
trainer.test(model, datamodule=dm)

# Make predictions on test set
dm = FileDataModule(f"data/{testdata}", model_name, batch_size=8)
trainer.test(model, datamodule=dm)
