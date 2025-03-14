# pyvers
A Python package for claim verification

## Description

Claim verification using Hugging Face transformers and PyTorch Lightning.
Supported datasets are Fever, SciFact, and a very small and Easy handmade one.


## Usage

This example takes about a minute on a CPU.

```python
# Import required modules
import pytorch_lightning as pl
from pyvers.data import EasyDataModule
from pyvers.model import PyversClassifier

# Instantiate data and model
dm = EasyDataModule("bert-base-uncased")
model = PyversClassifier(dm.model_name)

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

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         Accuracy          │           89.0            │
│          F1_NEI           │           80.0            │
│         F1_REFUTE         │           100.0           │
│        F1_SUPPORT         │           86.0            │
└───────────────────────────┴───────────────────────────┘

[['SUPPORT', 'SUPPORT', 'SUPPORT', 'NEI', 'NEI', 'SUPPORT', 'REFUTE', 'REFUTE', 'REFUTE']]

# Ground-truth labels are:
# [['SUPPORT', 'SUPPORT', 'SUPPORT', 'NEI', 'NEI', 'NEI', 'REFUTE', 'REFUTE', 'REFUTE']]
```

## Changing the model; Zero-shot predictions

The label-to-id mapping in pyvers is `{"SUPPORTS":0, "NOT ENOUGH INFO":1, "REFUTES":2}`.
The IDs match the ordering of label names (`["entailment", "neutral", "contradiction"]`) in a [DeBERTa model trained on MultiNLI, Fever-NLI and Adversarial-NLI (ANLI)](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli), so we can use this model for zero-shot classification of claim-evidence pairs.

```
dm = EasyDataModule("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
model = PyversClassifier(dm.model_name)
trainer = pl.Trainer()
dm.setup(stage="test")
predictions = trainer.predict(model, datamodule=dm)
print(predictions)
# [['SUPPORT', 'SUPPORT', 'SUPPORT', 'REFUTE', 'REFUTE', 'REFUTE', 'REFUTE', 'REFUTE', 'REFUTE']]
```

This model misses NEIs but can distinguish between SUPPORT and REFUTE without fine-tuning on the Easy dataset.
