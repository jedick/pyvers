# Plot token sequence lengths for two datasets

from pyvers.data import FileDataModule
import numpy as np
import matplotlib.pyplot as plt

# Create a figure with a 1x2 grid of subplots
fig, axs = plt.subplots(1, 2, figsize=(8, 4))

datasets = ["scifact", "citint"]
titles = ["SciFact", "CitInt"]
for i in range(len(datasets)):
    dm = FileDataModule(f"data/{datasets[i]}", "bert-base-uncased", max_length=1024)
    dm.setup("fit")
    dl = iter(dm.train_dataloader())
    # Place to hold sequence lengths
    seq_len = np.array([])
    for data_tensor in dl:
        # Convert tensor to np array
        input_ids = data_tensor["input_ids"].numpy()
        # Number of non-zero ids in each row
        sequence_length = np.sum(input_ids != 0, axis=1)
        seq_len = np.concatenate((seq_len, sequence_length), axis=0)

    # Plot histogram
    axs[i].hist(seq_len)
    axs[i].set_title(f"{titles[i]}")
    axs[i].set_xlabel("Sequence length (tokens)")
    axs[i].set_ylabel("Count")

# Display the plots
plt.tight_layout()
plt.show()
