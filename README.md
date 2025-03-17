# pyvers
A Python package for claim verification

## Description

Claim verification using Hugging Face transformers and PyTorch Lightning.
Supported datasets are [Fever](https://huggingface.co/datasets/fever/fever), [SciFact](https://github.com/allenai/scifact), [Citation-Integrity](https://github.com/ScienceNLP-Lab/Citation-Integrity/), and a small toy dataset.

## Usage

This example takes about a minute on a CPU.

```python
# Import required modules
import pytorch_lightning as pl
from pyvers.data import ToyDataModule
from pyvers.model import PyversClassifier

# Initialize data and model
dm = ToyDataModule("bert-base-uncased")
model = PyversClassifier(dm.model_name)

# Train model
trainer = pl.Trainer(enable_checkpointing=False, max_epochs=20)
trainer.fit(model, datamodule=dm)

# Test model
trainer.test(model, datamodule=dm)

# Show predictions
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

## Loading data

Use one of the classes in `pyvers.data` to load data from files or from HuggingFace datasets (for Fever).
The format for data files is described [here](https://github.com/dwadden/multivers/blob/main/doc/data.md).
Get data files for SciFact and Citation-Integrity with labels used in pyvers [here](https://github.com/jedick/RefSup/tree/main/data).

```python
# This is the model used for the tokenizer
model_name = "bert-base-uncased"

# Read Fever data
from pyvers.data import FeverDataModule
dm = FeverDataModule(model_name)

# Read data from jsonl files
from pyvers.data import FileDataModule
dm = FileDataModule(model_name, "data/scifact")
```

We can also load data from two datasets for training on shuffled data.

```python
# Shuffle training data from SciFact and Citation-Integrity
dm = FileDataModule(model_name, ["data/scifact", "data/citint"])

# Take a look at the tokenized data ...
dm.setup("fit")
next(iter(dm.train_dataloader()))
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
dm = ToyDataModule("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
model = PyversClassifier(dm.model_name)
trainer = pl.Trainer()
dm.setup(stage="test")
predictions = trainer.predict(model, datamodule=dm)
print(predictions)
# [['SUPPORT', 'SUPPORT', 'SUPPORT', 'REFUTE', 'REFUTE', 'REFUTE', 'REFUTE', 'REFUTE', 'REFUTE']]
```

The pretrained model successfully distinguishes between SUPPORT and REFUTE on the toy dataset but misclassifies NEI as REFUTE.
This can be improved with fine-tuning.
