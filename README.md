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

## Label to ID mapping

| ID | pyvers  | [Fever](https://huggingface.co/datasets/fever/fever)* | [MultiNLI](https://huggingface.co/datasets/nyu-mll/multi_nli), [ANLI](https://huggingface.co/datasets/facebook/anli) |
| - | - | - | - |
| 0  | SUPPORT | SUPPORTS        | entailment |
| 1  | NEI     | NOT ENOUGH INFO | neutral |
| 2  | REFUTE  | REFUTES         | contradiction |

\* Text labels only

## Changing the model

This uses a [DeBERTa model trained on MultiNLI, Fever-NLI and Adversarial-NLI (ANLI)](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli) for zero-shot classification of claim-evidence pairs.
For other models, check that the mapping between labels and IDs is the same.

```
dm = EasyDataModule("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
model = PyversClassifier(dm.model_name)
trainer = pl.Trainer()
dm.setup(stage="test")
predictions = trainer.predict(model, datamodule=dm)
print(predictions)
# [['SUPPORT', 'SUPPORT', 'SUPPORT', 'REFUTE', 'REFUTE', 'REFUTE', 'REFUTE', 'REFUTE', 'REFUTE']]
```

The pretrained model successfully distinguishes between SUPPORT and REFUTE on the Easy dataset but misclassifies NEI as REFUTE.
This can be improved with fine-tuning.
