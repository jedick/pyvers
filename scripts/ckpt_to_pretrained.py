from pyvers.model import PyversClassifier
from transformers import AutoTokenizer

# Save checkpoint file saved by pl.trainer.save_checkpoint()
# as "pretrained" directory for loading with AutoModelForSequenceClassification.from_pretrained
model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
model = PyversClassifier.load_from_checkpoint(
    "/home/pyvers/checkpoints/shuffle3/shuffle_512_10epochs.ckpt"
)
model.model.save_pretrained(
    "/home/pyvers/DeBERTa-v3-base-mnli-fever-anli-scifact-citint"
)
# Also need to save the tokenizer
model_name = model.model.config._name_or_path
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained("/home/pyvers/DeBERTa-v3-base-mnli-fever-anli-scifact-citint")
