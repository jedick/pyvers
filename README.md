[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# pyvers

Python package for data processing and training of claim verification models.
This package was developed as part of an [ML engineering capstone project](https://github.com/jedick/MLE-capstone-project).

Claim verification is a task in natural language processing (NLP) with applications ranging from fact-checking to verifying the accuracy of scientific citations.
The models used in this package are based on the transformer deep-learning architecture.

## Features

- Data Modules
	- Support for local files and [HuggingFace datasets](https://huggingface.co/docs/hub/en/datasets).
	- Consistent label encoding for different natural language inference (NLI) datasets (see [below](#label-to-id-mapping)).
	- Supports [shuffling training data](https://github.com/jedick/pyvers/blob/main/scripts/shuffle_datasets.py) from multiple datasets for [improved model generalization](https://jedick.github.io/blog/experimenting-with-transformer-models-for-citation-verification/#cross-dataset-generalization).
- Trainer
 	- Training and data modules implemented with [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning).
    - Use any [pretrained sequence classification model](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForSequenceClassification) from HuggingFace.
    - Logger is configured to plot training and validation loss on the [same graph in TensorBoard](https://jedick.github.io/blog/experimenting-with-transformer-models-for-citation-verification/#the-paradox-of-rising-loss-and-improving-accuracy).

## Installation

Run these commands in the root directory of the repository.
- The first command installs the requirements.
- The second command installs the pyvers package in [development mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html).
  - Remove the `-e` for a standard installation.

```
pip install -r requirements.txt
pip install -e .
```

## Loading data

### `pyvers.data.FileDataModule`

- This class loads data from local data files in JSON lines format (jsonl).
- Supported datasets include [SciFact](https://github.com/allenai/scifact) and [Citation-Integrity](https://github.com/ScienceNLP-Lab/Citation-Integrity/).
- The schema for the data files is described [here](https://github.com/dwadden/multivers/blob/main/doc/data.md).
- Get data files for SciFact and Citation-Integrity with labels used in pyvers [here](https://github.com/jedick/MLE-capstone-project/tree/main/data).
- The data module can be used to shuffle training data from both datasets.

```python
from pyvers.data import FileDataModule
# Set the model used for the tokenizer
model_name = "bert-base-uncased"

# Load data from one dataset
dm = FileDataModule("data/scifact", model_name)

# Shuffle training data from two datasets
dm = FileDataModule(["data/scifact", "data/citint"], model_name)

# Get some tokenized data
dm.setup("fit")
next(iter(dm.train_dataloader()))
```

### `pyvers.data.NLIDataModule`

- This class loads data from selected HuggingFace datasets.
- Supported datasets are
[copenlu/fever_gold_evidence](https://huggingface.co/datasets/copenlu/fever_gold_evidence),
[facebook/anli](https://huggingface.co/datasets/facebook/anli), and
[nyu-mll/multi_nli](https://huggingface.co/datasets/nyu-mll/multi_nli).

```python
from pyvers.data import NLIDataModule
model_name = "bert-base-uncased"

# Load data from HuggingFace datasets
dm = NLIDataModule("facebook/anli", model_name)

# Get some tokenized data
dm.prepare_data()
dm.setup("fit")
next(iter(dm.train_dataloader()))
```

### `pyvers.data.ToyDataModule`

- This is a small handmade toy dataset.
- There are no data files; the dataset is hard-coded in the class definition.

## Fine-tuning example

This takes about a minute on a CPU.

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
│        AUROC Macro        │          0.963            │
│      AUROC Weighted       │          0.963            │
│         Accuracy          │           88.9            │
│         F1 Macro          │           88.6            │
│         F1 Micro          │           88.9            │
│          F1_NEI           │          100.0            │
│         F1_REFUTE         │           80.0            │
│        F1_SUPPORT         │           85.7            │
└───────────────────────────┴───────────────────────────┘

[['SUPPORT', 'SUPPORT', 'SUPPORT', 'NEI', 'NEI', 'NEI', 'REFUTE', 'REFUTE', 'SUPPORT']]

# Ground-truth labels are:
# [['SUPPORT', 'SUPPORT', 'SUPPORT', 'NEI', 'NEI', 'NEI', 'REFUTE', 'REFUTE', 'REFUTE']]
```

## Zero-shot example

This uses a [DeBERTa model trained on MultiNLI, Fever-NLI and Adversarial-NLI (ANLI)](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli) for zero-shot classification of claim-evidence pairs.

```
import pytorch_lightning as pl
from pyvers.model import PyversClassifier
from pyvers.data import ToyDataModule
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

*When using a pre-trained model for zero-shot classification, check the mapping between labels and IDs.*

```python
from transformers import AutoConfig

model_name = "answerdotai/ModernBERT-base"
config = AutoConfig.from_pretrained(model_name, num_labels=3)
print(config.to_dict()["id2label"])
# {0: 'LABEL_0', 1: 'LABEL_1', 2: 'LABEL_2'}

model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
config = AutoConfig.from_pretrained(model_name, num_labels=3)
print(config.to_dict()["id2label"])
# {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
```

Because it uses labels that are consistent with the NLI categories listed below, for *zero-shot classification* we would choose the pretrained DeBERTa model rather than ModernBERT.
However, *fine-tuning* either model for text classification should work (see [this page](https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/fine-tune-modern-bert-in-2025.ipynb) for information on fine-tuning ModernBERT).

## Label to ID mapping

| ID | pyvers  | [Fever](https://huggingface.co/datasets/fever/fever)* | [MultiNLI](https://huggingface.co/datasets/nyu-mll/multi_nli), [ANLI](https://huggingface.co/datasets/facebook/anli) |
| - | - | - | - |
| 0  | SUPPORT | SUPPORTS        | entailment |
| 1  | NEI     | NOT ENOUGH INFO | neutral |
| 2  | REFUTE  | REFUTES         | contradiction |

\* Text labels only

