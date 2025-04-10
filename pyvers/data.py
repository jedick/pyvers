import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from multivers.data_train import SciFactReader
# For reading data from jsonl files
from .verisci import SciFactReader
# HuggingFace datasets
import datasets

# Label IDs are consistent with this pre-trained model:
#     https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
#     label_names = ["entailment", "neutral", "contradiction"]
# SciFact: label2id = {"SUPPORT":0, "NEI":1, "REFUTE":2}
# Fever:   label2id = {"SUPPORTS":0, "NOT ENOUGH INFO":1, "REFUTES":2}

class PyversDataset(Dataset):
    def __init__(self, dataset, model_name, max_length):
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset["claims"])

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            # Tokenize a sequence pair as follows:
            # [CLS] {claim tokens} [SEP] {evidence tokens} [SEP]
            # NOTE: evidence (premise) should come before claims (hypothesis)
            # - Gives better zero-shot predictions on the toy dataset (see README.md) and SciFact
            self.dataset["evidences"][idx],
            self.dataset["claims"][idx],
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
            truncation="longest_first",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "token_type_ids": encoding["token_type_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.dataset["labels"][idx], dtype=torch.long),
        }

class FileDataModule(pl.LightningDataModule): 
    def __init__(self, model_name, directory="data/scifact", batch_size=32, max_length=512, use_val_for_test=False): 
        super().__init__() 
        self.model_name = model_name
        self.directory = directory
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_val_for_test = use_val_for_test
        self.num_workers = 4
          
    def prepare_data(self): 
        pass

    def setup(self, stage=None): 

        @staticmethod
        def get_one_dataset(fold, directory):
            # Instantiate the data reader
            reader = SciFactReader(directory)
            data = reader.get_text_data(fold)
            # Process data to get claims and evidence sentences
            claims = [item["to_tensorize"]["claim"] for item in data]
            evidences = [" ".join(item["to_tensorize"]["sentences"]) for item in data]
            # Process data to get labels
            labels = [item["to_tensorize"]["label"] for item in data]
            label2id = {"SUPPORT":0, "NEI":1, "REFUTE":2}
            ids = [label2id[label] for label in labels]
            return dict(evidences=evidences, claims=claims, labels=ids)

        @staticmethod
        def get_data(fold, directory):
            # Put a single directory name into a list so we can iterate over it
            if isinstance(directory, str):
                directory = [directory]
            # Get the data from one or more directories, as a list of dictionaries
            all_data = [get_one_dataset(fold, dir) for dir in directory]
            # Use dict comprehension to combine the lists for each dictionary key
            all_data = {k: [d[k] for d in all_data] for k in all_data[0]}
            # Use list comprehension to flatten the lists for each dictionary key
            all_data = {k: [item for sublist in all_data[k] for item in sublist] for k in all_data}
            return all_data

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            # Prepare data
            train_data = get_data("train", self.directory)
            self.train_dataset = PyversDataset(train_data, self.model_name, self.max_length)
            # The validation set is called the dev set in the SciFact paper
            val_data = get_data("dev", self.directory)
            self.val_dataset = PyversDataset(val_data, self.model_name, self.max_length)

        # Assign test dataset for use in dataloader
        if stage == "test":
            # If labels for the test set aren't available, we can calculate metrics with the validation set instead
            if(self.use_val_for_test):
                test_data = get_data("dev", self.directory)
            else:
                test_data = get_data("test", self.directory)
            
            self.test_dataset = PyversDataset(test_data, self.model_name, self.max_length)

    def train_dataloader(self): 
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
  
    def val_dataloader(self): 
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
  
    def test_dataloader(self): 
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self): 
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

class ToyDataModule(pl.LightningDataModule): 
    def __init__(self, model_name, batch_size=32, max_length=32): 
        super().__init__() 
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = 4
          
    def prepare_data(self): 
        pass

    def setup(self, stage=None): 

        # Assign train/val datasets for use in dataloaders
        train_data = {
            "evidences": [
                # SUPPORT
                "All plants are red.", "All plants are green.", "All plants are blue.", "All cars are red.",
                # NEI
                "All cars are red.", "All cars are green.", "All cars are blue.",
                "All cars are red.", "All cars are green.", "All cars are blue.",
                # REFUTE
                "All plants are red.", "All plants are green.", "All plants are blue.", "All cars are red.",
            ],
            "claims": [
                # SUPPORT
                "This plant is red.", "This plant is green.", "This plant is blue.", "This car is red.",
                # NEI
                "This plant is green.", "This plant is blue.", "This plant is red.",
                "This plant is red.", "This plant is green.", "This plant is blue.",
                # REFUTE
                "This plant is green.", "This plant is blue.", "This plant is red.", "This car is green.",
            ],
            "labels": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]
        }
        self.train_dataset = PyversDataset(train_data, self.model_name, max_length=self.max_length)

        test_data = {
            "evidences": [
                # SUPPORT
                "All plants are red.", "All plants are yellow.", "All cars are yellow.",
                # NEI
                "All cars are red.", "All cars are red.", "All plants are yellow.",
                # REFUTE
                "All plants are red.", "All plants are red.", "All cars are red.",
            ],
            "claims": [
                # SUPPORT
                "This plant is red.", "This plant is yellow.", "This car is yellow.",
                # NEI
                "This plant is blue.", "This plant is yellow.", "This car is yellow.",
                # REFUTE
                "This plant is blue.", "This plant is yellow.", "This car is yellow.",
            ],
            "labels": [0, 0, 0, 1, 1, 1, 2, 2, 2]
        }
        self.test_dataset = PyversDataset(test_data, self.model_name, max_length=self.max_length)

    def train_dataloader(self): 
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
  
    def val_dataloader(self): 
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
  
    def test_dataloader(self): 
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self): 
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

class NLIDataModule(pl.LightningDataModule):
    """
    Data module for NLI datasets - Fever, ANLI, and MNLI

    dataset_name: HuggingFace dataset name, one of "copenlu/fever_gold_evidence", "facebook/anli"
    model_name: HuggingFace model name, defaults to "bert-base-uncased"
    batch_size: Batch size for data loader
    max_length: Maximum sequence length for tokenizer
    """
    def __init__(self, dataset_name="facebook/anli", model_name="bert-base-uncased", batch_size=32, max_length=128):
        super().__init__()
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = 4

    def prepare_data(self):
        # Download the dataset and tokenizer
        datasets.load_dataset(self.dataset_name)
        AutoTokenizer.from_pretrained(self.model_name)

    def setup(self, stage=None):
        # Load the dataset
        dataset = datasets.load_dataset(self.dataset_name)

        if stage == "fit":
            train_data = self.get_data(self.dataset_name, dataset, "train")
            self.train_dataset = PyversDataset(train_data, self.model_name, self.max_length)
            val_data = self.get_data(self.dataset_name, dataset, "val")
            self.val_dataset = PyversDataset(val_data, self.model_name, self.max_length)

        if stage == "test":
            test_data = self.get_data(self.dataset_name, dataset, "test")
            self.test_dataset = PyversDataset(test_data, self.model_name, self.max_length)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    @staticmethod
    def get_data(dataset_name, dataset, fold):
        """
        Extracts data in consistent format from various datasets.
        Uses fold (train, val or test) to extract specific split from dataset.
        Uses same label encoding for all datasets.

        dataset_name: HuggingFace dataset name
        dataset: Dataset returned by datasets.load_dataset()
        fold: Name of the fold used in PyTorch Lightning (train, val, or test)
        """
        if dataset_name == "copenlu/fever_gold_evidence":
            if fold == "train":
                split = dataset["train"]
            if fold == "val":
                split = dataset["validation"]
            if fold == "test":
                split = dataset["test"]
            claims = split["claim"]
            # The evidence list includes the page title, but we just want the evidence sentences
            evidences = [item[0][2] for item in split["evidence"]]
            label2id = {"SUPPORTS":0, "NOT ENOUGH INFO":1, "REFUTES":2}
            labels = [label2id[label] for label in split["label"]]
            return {"claims":claims, "evidences":evidences, "labels":labels}
        if dataset_name == "facebook/anli":
            if fold == "train":
                split = dataset["train_r3"]
            if fold == "val":
                split = dataset["dev_r3"]
            if fold == "test":
                split = dataset["test_r3"]
            claims = split["hypothesis"]
            evidences = split["premise"]
            labels = split["label"]
            return {"claims":claims, "evidences":evidences, "labels":labels}

