# pyvers
A Python package for claim verification

## Description

Claim verification using Hugging Face transformers and PyTorch Lightning.
Supported training datasets are Fever, SciFact, and a very small and Easy handmade one.

## Usage

This example takes about a minute on a CPU.

```python
# Import required modules
import pytorch_lightning as pl
from pyvers.data import EasyDataModule
from pyvers.model import BERTClassifier

# Instantiate data and model
dm = EasyDataModule("bert-base-uncased")
model = BERTClassifier()

# Train model
trainer = pl.Trainer(enable_checkpointing=False, max_epochs=20)
trainer.fit(model, datamodule=dm)

# Test model
trainer.test(model, datamodule=dm)

# Show the predictions
predictions = trainer.predict(model, datamodule=dm)
print(predictions)
```

This is what we get (results vary between runs):

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         Accuracy          │           89.0            │
│          F1_NEI           │           80.0            │
│         F1_REFUTE         │           100.0           │
│        F1_SUPPORT         │           86.0            │
└───────────────────────────┴───────────────────────────┘

[['REFUTE', 'REFUTE', 'REFUTE', 'NEI', 'NEI', 'SUPPORT', 'SUPPORT', 'SUPPORT', 'SUPPORT']]
