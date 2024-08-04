from model import *

import torch.optim as optim
import numpy as np

vocab = {
    "[PAD]": 0,
    "A": 1,
    "C": 2,
    "G": 3,
    "T": 4,
    "-": 5,
    "[SEP]": 6,
    "[CLS]": 7,
    "[EOS]": 8,
}


def input_tokenization(seq_1: str, seq_2: str, max_length: int, vocab: dict):
    """
    Encoding inputs for Encoder-only Transformer.
    e.g., INPUT, previous bases: AACCTTTT; current base: T
          ENCODED INPUT: [CLS]AACCTTTT[SEP]T[SEP]NNNN...
    """
    array = (
        [vocab["[CLS]"]]
        + [vocab[char] for char in seq_1]
        + [vocab["[SEP]"]]
        + [vocab[char] for char in seq_2]
        + [vocab["[SEP]"]]
    )
    while len(array) < max_length:
        # padding 0 for remain bases
        array.append(vocab["[PAD]"])
    array = np.array(array, dtype=np.float32)
    return array


def label_tokenization(seq: str, vocab: dict):
    array = [vocab["[CLS]"]] + [vocab[char] for char in seq] + [vocab["[EOS]"]]
    array = np.array(array, dtype=np.float32)
    return array


num_bases = 8  # [PAD], A, C, G, T, [CLS], [SEP], [EOS]

embed_size = 128
num_layers = 1
forward_expansion = 4
heads = 8
max_length = 250

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ErrorPrediction(
    embed_size, heads, num_layers, forward_expansion, num_bases, 0.1, max_length
)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss(ignore_index=num_bases - 1)

input_array = input_tokenization("CTGACATGCACAC", "A", 250, vocab)
label_array = label_tokenization("GGG", vocab)
input_indices = torch.tensor(input_array).to(device)  # Example input
label_indices = torch.tensor(label_array).to(device)
print(input_indices, label_indices)

# Train loop
for epoch in range(10):
    optimizer.zero_grad()
    output = model(input_indices)

    # Reshaping for calculating the loss
    output = output.view(-1, num_bases)
    output_target = label_indices.view(-1)
    loss = criterion(output, output_target)
    loss.backward()
    optimizer.step()
