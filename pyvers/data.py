import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from multivers.data_train import SciFactReader
# HuggingFace datasets
import datasets

class PyverDataset(Dataset):
    def __init__(self, dataset, tokenizer_name, max_length=128):
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset["claims"])

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            # Tokenize a sequence pair as follows:
            # [CLS] {evidence tokens} [SEP] {claim tokens} [SEP]
            self.dataset["evidences"][idx],
            self.dataset["claims"][idx],
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
            # Only truncate the evidence, not the claim
            truncation="only_first",
        )
        return {
            # Try this instead?
            #'input_ids': encoding['input_ids'].squeeze(),
            #'attention_mask': encoding['attention_mask'].squeeze(),
            #'labels': torch.tensor(item['label'])
            "input_ids": encoding["input_ids"].flatten(),
            "token_type_ids": encoding["token_type_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.dataset["labels"][idx], dtype=torch.long)
        }

class SciFactDataModule(pl.LightningDataModule): 
    def __init__(self, tokenizer_name, batch_size=32, max_length=512): 
        super().__init__() 
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = 4
          
    def prepare_data(self): 
        pass

    def setup(self, stage=None): 

        def get_data(split):
            # Instantiate the data reader
            reader = SciFactReader("data/scifact")
            data = reader.get_text_data(split)
            # Process data to get claims and evidence sentences
            claims = [item["to_tensorize"]["claim"] for item in data]
            evidences = [" ".join(item["to_tensorize"]["sentences"]) for item in data]
            # Process data to get labels
            labels = [item["to_tensorize"]["label"] for item in data]
            label2id = {"REFUTE":0, "NEI":1, "SUPPORT":2}
            ids = [label2id[label] for label in labels]
            return dict(evidences=evidences, claims=claims, labels=ids)

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            # Prepare data
            train_data = get_data("train")
            self.train_dataset = PyverDataset(train_data, self.tokenizer_name, self.max_length)
            val_data = get_data("val")
            self.val_dataset = PyverDataset(val_data, self.tokenizer_name, self.max_length)

        # Assign test dataset for use in dataloader
        if stage == "test":
            test_data = get_data("test")
            self.test_dataset = PyverDataset(test_data, self.tokenizer_name, self.max_length)

    def train_dataloader(self): 
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
  
    def val_dataloader(self): 
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
  
    def test_dataloader(self): 
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self): 
        # Make all predictions in a single batch
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers)

class EasyDataModule(pl.LightningDataModule): 
    def __init__(self, tokenizer_name, batch_size=32, max_length=32): 
        super().__init__() 
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = 4
          
    def prepare_data(self): 
        pass

    def setup(self, stage=None): 

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            train_data = {
                "evidences": [
                    # REFUTE
                    "All plants are red.", "All plants are green.", "All plants are blue.", "All cars are red.",
                    # NEI
                    "All cars are red.", "All cars are green.", "All cars are blue.",
                    "All cars are red.", "All cars are green.", "All cars are blue.",
                    # SUPPORT
                    "All plants are red.", "All plants are green.", "All plants are blue.", "All cars are red.",
                ],
                "claims": [
                    # REFUTE
                    "This plant is green.", "This plant is blue.", "This plant is red.", "This car is green.",
                    # NEI
                    "This plant is green.", "This plant is blue.", "This plant is red.",
                    "This plant is red.", "This plant is green.", "This plant is blue.",
                    # SUPPORT
                    "This plant is red.", "This plant is green.", "This plant is blue.", "This car is red.",
                ],
                "labels": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]
            }
            self.train_dataset = PyverDataset(train_data, self.tokenizer_name, max_length=self.max_length)

        # Assign test dataset for use in dataloader
        if stage == "test":
            test_data = {
                "evidences": [
                    # REFUTE
                    "All plants are red.", "All plants are red.", "All cars are red.",
                    # NEI
                    "All cars are red.", "All cars are red.", "All plants are yellow.",
                    # SUPPORT
                    "All plants are red.", "All plants are yellow.", "All cars are yellow.",
                ],
                "claims": [
                    # REFUTE
                    "This plant is blue.", "This plant is yellow.", "This car is yellow.",
                    # NEI
                    "This plant is blue.", "This plant is yellow.", "This car is yellow.",
                    # SUPPORT
                    "This plant is red.", "This plant is yellow.", "This car is yellow.",
                ],
                "labels": [0, 0, 0, 1, 1, 1, 2, 2, 2]
            }
            self.test_dataset = PyverDataset(test_data, self.tokenizer_name, max_length=self.max_length)

    def train_dataloader(self): 
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
  
    def test_dataloader(self): 
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self): 
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

class FeverDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer_name, batch_size=32, max_length=128):
        super().__init__()
        self.dataset_name = "copenlu/fever_gold_evidence"
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = 4

    def prepare_data(self):
        # Download the dataset
        datasets.load_dataset(self.dataset_name)

    def setup(self, stage=None):
        # Load the dataset
        dataset = datasets.load_dataset(self.dataset_name)

        def get_data(split):
            claims = split["claim"]
            # The evidence list includes the page title, but we just want the evidence sentences
            evidences = [item[0][2] for item in split["evidence"]]
            label2id = {"REFUTES":0, "NOT ENOUGH INFO":1, "SUPPORTS":2}
            labels = [label2id[label] for label in split["label"]]
            return {"claims":claims, "evidences":evidences, "labels":labels}

        if stage == "fit":
            train_data = get_data(dataset["train"])
            self.train_dataset = PyverDataset(train_data, self.tokenizer_name, self.max_length)
            val_data = get_data(dataset["validation"])
            self.val_dataset = PyverDataset(val_data, self.tokenizer_name, self.max_length)

        if stage == "test":
            test_data = get_data(dataset["test"])
            self.test_dataset = PyverDataset(test_data, self.tokenizer_name, self.max_length)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

