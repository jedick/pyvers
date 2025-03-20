# pyvers
A Python package for claim verification using Hugging Face transformers and PyTorch Lightning.
Supported datasets are [Fever](https://huggingface.co/datasets/fever/fever), [SciFact](https://github.com/allenai/scifact), [Citation-Integrity](https://github.com/ScienceNLP-Lab/Citation-Integrity/), and a small toy dataset.

## Features

- The trainer logs metrics to different directories allowing to plot training and validation loss on the [same graph in TensorBoard](https://jedick.github.io/blog/experimenting-with-transformer-models-for-citation-verification/#the-paradox-of-rising-loss-and-improving-accuracy).
- Ability to [pass multiple dataset directories](https://github.com/jedick/pyvers/blob/main/scripts/shuffle_datasets.py) to Lightning data module class to shuffle training data and [improve generalization performance across datasets](https://jedick.github.io/blog/experimenting-with-transformer-models-for-citation-verification/#cross-dataset-generalization).
- Use any pretrained model available for [AutoModelForSequenceClassification](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForSequenceClassification) from HuggingFace.

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
However, *fine-tuning* either model for text classification should work (see [this page for ModernBERT](https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/fine-tune-modern-bert-in-2025.ipynb)).

## Label to ID mapping

| ID | pyvers  | [Fever](https://huggingface.co/datasets/fever/fever)* | [MultiNLI](https://huggingface.co/datasets/nyu-mll/multi_nli), [ANLI](https://huggingface.co/datasets/facebook/anli) |
| - | - | - | - |
| 0  | SUPPORT | SUPPORTS        | entailment |
| 1  | NEI     | NOT ENOUGH INFO | neutral |
| 2  | REFUTE  | REFUTES         | contradiction |

\* Text labels only

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



