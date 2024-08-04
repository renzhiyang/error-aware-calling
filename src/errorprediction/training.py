import os
from turtle import forward
import hydra
import gzip

import numpy as np
import torch.optim as optim
import data_loader_training as data

from model import *
from typing import List, Dict
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


class DataEncoder_readbased:
    """
    Encoder for the intermediate output file produced by label_data.py
    编码时加入了[PAD], [SEP], [CLS], [EOS]等token, 用于BERT-like model, 输入是仅有read没有alignment的cigar信息
    并且这里对于insertion 采用递归式的编码, 例如 AACCT(TTT) 3-T insertion
    编码成三个输入，分别是
    input1: [CLS]AACCT[SEP]T  label: -
    input2: [CLS]ACCT-[SEP]T, label: -
    input3: [CLS]CCT--[SEP]T, label: -
    """

    def __init__(self, file_path, config, chunk_size=1000):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.line_offsets = []
        self.total_lines = 0
        self.config = config
        self.vocab = {
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

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist")

        with open(file_path, "r") as file:
            offset = 0
            for line in file:
                if self.total_lines % self.chunk_size == 0:
                    self.line_offsets.append(offset)
                offset += len(line)
                self.total_lines += 1

    def __len__(self):
        return self.total_lines

    def __getitem__(self, idx):
        chunk_idx = idx // self.chunk_size
        line_idx = idx % self.chunk_size

        with open(self.file_path, "r") as file:
            file.seek(self.line_offsets[chunk_idx])
            for _ in range(line_idx):
                file.readline()

            line = file.readline()
            parts = line.strip().split(" ")

            position = parts[2].split(":")[1]
            sequence_around = parts[-1].split(":")[1]
            variant_type = parts[-2].split(":")[1]
            read_base = parts[4].split(":")[1]
            truth_base = parts[3].split(":")[1]
            forward_bases = sequence_around[0 : self.config.label.window_size_half]
            behind_bases = sequence_around[-self.config.label.window_size_half :]

            if variant_type == "Insertion":
                results = []
                for i in range(len(read_base)):
                    update_forward_bases = forward_bases + (i * "-")
                    input_array = input_tokenization(
                        update_forward_bases,
                        read_base[i],
                        self.config.training.input_length,
                        self.vocab,
                    )
                    label_array = label_tokenization("-", self.vocab)
                    results.append((input_array, label_array))
                    # print test
                    # print(f'pos: {position}, Insertion forward base: {update_forward_bases}, read_base:{read_base[i]},'
                    #      f'truth_seq: -, len_input: {len(input_array)}\n')
                return results
            else:
                if variant_type == "Deletion":
                    forward_bases = forward_bases[:-1]

                input_array = input_tokenization(
                    forward_bases,
                    read_base,
                    self.config.training.input_length,
                    self.vocab,
                )
                label_array = label_tokenization(truth_base, self.vocab)
                # print test
                # print(f'pos: {position}, {variant_type} forward base: {forward_bases}, len_input:{len(input_array)}'
                #      f'read_base:{read_base}, label:{truth_base}\n')
                return input_array, label_array


class DataEncoder_cigarbased(DataEncoder_readbased):
    """
    Encoder for the intermediate output file produced by label_data.py
    输入仍然是read, 但是附带有alignment的cigar信息。
    """

    def __getitem__(self, idx):
        chunk_idx = idx // self.chunk_size
        line_idx = idx % self.chunk_size

        with open(self.file_path, "r") as file:
            file.seek(self.line_offsets[chunk_idx])
            for _ in range(line_idx):
                file.readline()

            line = file.readline()
            parts = line.strip().split(" ")

            position = parts[2].split(":")[1]
            sequence_around = parts[-1].split(":")[1]
            variant_type = parts[-2].split(":")[1]
            label_type = parts[1].split(":")[1]
            read_base = parts[4].split(":")[1]
            truth_base = parts[3].split(":")[1]
            forward_bases = sequence_around[0 : self.config.label.window_size_half]
            behind_bases = sequence_around[-self.config.label.window_size_half :]

            if variant_type == "SNV":
                input_array = input_tokenization(
                    forward_bases,
                    read_base,
                    self.config.training.input_length,
                    self.vocab,
                )
                label_array = label_tokenization_fix_length(
                    truth_base, self.config.training.label_length, self.vocab
                )
                # return input_array, label_array
            elif variant_type == "Insertion":
                input_array = input_tokenization(
                    forward_bases[:-1],
                    forward_bases[-1] + read_base,
                    self.config.training.input_length,
                    self.vocab,
                )
                label_seq = forward_bases[-1] + "-" * len(read_base)
                label_array = label_tokenization_fix_length(
                    label_seq, self.config.training.label_length, self.vocab
                )
                # return input_array, label_array
            else:
                input_array = input_tokenization(
                    forward_bases[:-1],
                    read_base,
                    self.config.training.input_length,
                    self.vocab,
                )
                label_array = label_tokenization_fix_length(
                    truth_base, self.config.training.label_length, self.vocab
                )

            if input_array is None:
                return None
            label_type = (
                np.array([1, 0]) if label_type == "Positive" else np.array([0, 1])
            )
            label_seq_ont_hot = ont_hot_encoding(label_array, 5)
            return input_array, label_type, label_seq_ont_hot


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

    if len(array) > max_length:
        return None

    while len(array) < max_length:
        # padding 0 for remain bases
        array.append(vocab["[PAD]"])

    array = np.array(array, dtype=np.float32)
    return array


def label_tokenization(seq: str, vocab: dict):
    array = [vocab["[CLS]"]] + [vocab[char] for char in seq] + [vocab["[EOS]"]]
    array = np.array(array, dtype=np.float32)
    return array


def label_tokenization_fix_length(seq: str, label_len: int, vocab: dict):
    array = [vocab[char] for char in seq]
    # TODO: 对于某些大于20的Indel 目前先截断 后面需要完善
    if len(array) > label_len:
        array = array[:label_len]

    while len(array) < label_len:
        array.append(vocab["[PAD]"])
    array = np.array(array, dtype=np.float32)
    return array


def ont_hot_encoding(label_array, num_class: int):
    """
    只对 A,C,G,T,-,[PAD] encoding
    A: [1, 0, 0, 0, 0], ..., [PAD]: [0, 0, 0, 0, 0]
    """
    output = []
    mapping = np.eye(num_class)
    for value in label_array:  # type: ignore
        if value == 0:
            output.append(np.zeros(num_class))
        else:
            output.append(np.eye(num_class)[int(value) - 1])
    output = np.array(output, dtype=np.float32)
    return output


def custon_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])
    inputs, label_types, labels = zip(*batch)
    inputs = torch.stack([torch.tensor(input, dtype=torch.float32) for input in inputs])
    label_types = torch.stack(
        [torch.tensor(label_type, dtype=torch.float32) for label_type in label_types]
    )
    labels = torch.stack([torch.tensor(label, dtype=torch.float32) for label in labels])
    return inputs, label_types, labels


def create_data_loader(dataset, batch_size: int, train_ratio: float):
    """
    Create DataLoader object for training
    """
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])  # type: ignore

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custon_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, collate_fn=custon_collate_fn
    )

    return train_loader, test_loader


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def train(model, train_loader, test_loader, criterion, optimizer, config):
    save_interval = 10
    epochs = config.training.epochs
    model_dir = config.training.model_path + "/" + start_time
    ensure_dir(model_dir)

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for i, (inputs, label_types, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, label_types, labels = (
                inputs.to(device),
                label_types.to(device),
                labels.to(device),
            )
            # print(label_types)
            # print(inputs, labels, inputs.shape, labels.shape)
            src_mask = inputs == 0
            src_mask = torch.permute(src_mask, (1, 0))
            src_mask = src_mask.to(device)
            # print(f'input: {inputs.shape}, mask:{src_mask.shape}, label:{labels.shape}')

            # with mask, produce nan output. https://github.com/pytorch/pytorch/issues/24816
            # outputs = model(inputs, src_mask)
            # without mask
            outputs = model(inputs)

            # print(f'output shape: {outputs.shape}, label shape: {labels.shape}')
            # print(f'outputs: {outputs}')
            # print(f'labels: {label_types}')
            # loss = criterion(outputs, labels)
            loss = criterion(outputs, label_types)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        writer.add_scalar("Loss/train", epoch_loss, epoch)

        val_loss, accuracy = test(model, test_loader, criterion)
        writer.add_scalar("Loss/test", val_loss, epoch)
        writer.add_scalar("Accuracy/test", accuracy, epoch)
        writer.flush()
        print(
            f"Time:{datetime.now()}, Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}",
            flush=True,
        )

        if (epoch + 1) % save_interval == 0:
            model_save_path = os.path.join(
                model_dir, f"{config.training.out_predix}_epoch_{epoch+1}.pt"
            )
            torch.save(model.state_dict(), model_save_path)
            model.load_state_dict(torch.load(model_save_path, map_location=device))
            print(f"Model saved at epoch {epoch+1}", flush=True)


def test(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, label_types, labels in test_loader:
            inputs, label_types, labels = (
                inputs.to(device),
                label_types.to(device),
                labels.to(device),
            )
            src_mask = inputs == 0
            src_mask = torch.permute(src_mask, (1, 0))
            src_mask = src_mask.to(device)

            outputs = model(inputs, src_mask)
            # loss = criterion(outputs, labels)
            loss = criterion(outputs, label_types)
            running_loss += loss.item()
            # print(f'labels 1: {labels}')
            # _, predicted = torch.max(outputs.data, 2)
            # _, labels = torch.max(labels.data, 2)
            _, predicted = torch.max(outputs.data, 1)
            _, label_types = torch.max(label_types.data, 1)
            # print(f'labels 2: {labels}, predicted: {predicted}')
            # total += labels.size(0)
            total += label_types.size(0)
            correct += (predicted == label_types).sum().item()
            # correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(test_loader)
    accuracy = 100 * correct / total
    return val_loss, accuracy


@hydra.main(
    version_base=None,
    config_path="../../configs/error_prediction",
    config_name="params.yaml",
)
def main(config: DictConfig) -> None:
    model = ErrorPrediction_with_CIGAR_onlyType(
        embed_size=config.training.embed_size,
        heads=config.training.heads,
        num_layers=config.training.num_layers,
        forward_expansion=config.training.forward_expansion,
        num_tokens=config.training.num_tokens,
        num_bases=config.training.num_bases,
        dropout_rate=config.training.dropout_rate,
        max_length=config.training.max_length,
        output_length=config.training.label_length,
    ).to(device)

    # output model structure
    onnx_input = torch.ones((40, 256))
    # print(f'pytorch version: {torch.__version__}', flush=True)
    # torch.onnx.export(model, onnx_input, 'model_2class.onnx',  # type: ignore
    #                  input_names=["input sequence"], output_names=["prediction"])
    # tensorboard visualize model
    # writer.add_graph(model, onnx_input)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)

    dataset = DataEncoder_cigarbased(
        file_path=config.data_path.label_f,
        config=config,
        chunk_size=config.training.data_loader_chunk_size,
    )
    train_loader, test_loader = create_data_loader(
        dataset,
        batch_size=config.training.batch_size,
        train_ratio=config.training.train_ratio,
    )

    train(model, train_loader, test_loader, criterion, optimizer, config)
    writer.close()


if __name__ == "__main__":
    torch.set_printoptions(threshold=10_000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(f"runs/experiment-{start_time}")
    print(device, flush=True)
    main()

