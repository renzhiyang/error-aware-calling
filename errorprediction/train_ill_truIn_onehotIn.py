import os
import hydra
import torch
import torch.nn as nn

import errorprediction.models.nets as nets
import errorprediction.utils as utils

# from errorprediction.model import *
from errorprediction.models.baseline import Baseline, Baseline_Kmer_In
from errorprediction.data_loader_truth_input_training import Data_Loader

from torchinfo import summary  # type: ignore
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # type: ignore


def custon_collate_fn(batch):
    if not batch:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])
    inputs, labels_1, labels_2 = zip(*batch)
    inputs = torch.stack([torch.tensor(input, dtype=torch.float32) for input in inputs])
    labels_1 = torch.stack(
        [torch.tensor(label_1, dtype=torch.float32) for label_1 in labels_1]
    )
    labels_2 = torch.stack(
        [torch.tensor(label_2, dtype=torch.float32) for label_2 in labels_2]
    )
    return inputs, labels_1, labels_2


def create_data_loader(dataset, batch_size: int, train_ratio: float):
    """
    Create DataLoader object for training
    dataset = [
        [(input, label1, label2), (input, label1, label2), ...],
        [(input, label1, label2), (input, label1, label2), ...],
        ...
    ]
    """
    print(f"original len: {len(dataset)}")
    dataset = [item for item in dataset if item is not None]
    flat_data = [
        sample for sublist in dataset for sample in sublist if sample is not None
    ]
    train_size = int(train_ratio * len(flat_data))
    test_size = len(flat_data) - train_size
    print(f"flat dataset len: {len(flat_data)}, train: {train_size}, test: {test_size}")
    train_dataset, test_dataset = random_split(flat_data, [train_size, test_size])  # type: ignore
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custon_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, collate_fn=custon_collate_fn
    )

    return train_loader, test_loader


def count_matching_indices(tensor1, tensor2):
    matching_indices = tensor1 == tensor2
    num_matching_indices = matching_indices.sum()
    return num_matching_indices


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def log_weights(epoch, writer, model):
    for name, param in model.named_parameters():
        # log the weights of each layer
        writer.add_histogram(f"{name}/weights", param, epoch)
        # log the gradients of each layer
        if param.grad is not None:
            writer.add_histogram(f"{name}/gradients", param.grad, epoch)


def train(
    model,
    train_loader,
    test_loader,
    criterion_1,
    criterion_2,
    optimizer,
    writer,
    config,
):
    save_interval = 10
    epochs = config.training.epochs
    model_dir = config.training.model_path + "/" + start_time
    ensure_dir(model_dir)

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for i, (inputs, labels_1, labels_2) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, labels_1, labels_2 = (
                inputs.to(device),
                labels_1.to(device),
                labels_2.to(device),
            )
            # print(f'num: {i+1}, input shape:{inputs.shape}')
            # print(f'label1 shape: {labels_1.shape}, label2 shape: {labels_2.shape}')
            next_base, insertion = model(inputs)
            # print(f"training next_base: {next_base}, training label: {labels_1}")
            # print(f"training insertion: {insertion}, training label: {labels_2}")

            loss_1 = criterion_1(next_base, labels_1)
            loss_2 = criterion_2(insertion, labels_2)
            loss = loss_1 + loss_2
            loss.backward()
            optimizer.step()
            running_loss += loss_1.item() + loss_2.item()

        epoch_loss = running_loss / len(train_loader)
        writer.add_scalar("Loss/train", epoch_loss, epoch)

        val_loss, accuracy = test(model, test_loader, criterion_1, criterion_2, config)
        writer.add_scalar("Loss/test", val_loss, epoch)
        writer.add_scalar("Accuracy/test", accuracy, epoch)
        log_weights(epoch, writer, model)
        writer.flush()
        print(
            f"Time:{datetime.now()}, Epoch {epoch+1}/{epochs}, Loss_train: {epoch_loss:.4f}, Loss_test: {val_loss:.4f}, Accuracy: {accuracy:.4f} \n",
            flush=True,
        )

        if (epoch + 1) % save_interval == 0:
            model_save_path = os.path.join(
                model_dir, f"{config.training.out_predix}_epoch_{epoch+1}.pt"
            )
            torch.save(model.state_dict(), model_save_path)
            # model.load_state_dict(torch.load(model_save_path, map_location=device))
            print(f"Model saved at epoch {epoch+1} \n", flush=True)


def test(model, test_loader, criterion_1, criterion_2, config):
    model.eval()
    running_loss = 0.0
    correct = 0
    next_base_correct = 0
    insertion_correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels_1, labels_2 in test_loader:
            inputs, labels_1, labels_2 = (
                inputs.to(device),
                labels_1.to(device),
                labels_2.to(device),
            )

            next_base_p, insertion_p = model(inputs)
            # loss = criterion(outputs, labels)
            loss_1 = criterion_1(next_base_p, labels_1)
            loss_2 = criterion_2(insertion_p, labels_2)
            loss = loss_1 + loss_2

            running_loss += loss.item()
            _, next_base = torch.max(next_base_p.data, 1)
            _, insertion = torch.max(insertion_p.data, 1)
            _, true_next_base = torch.max(labels_1.data, 1)
            _, true_insertion = torch.max(labels_2.data, 1)
            # print(f'next_base: {next_base_p}')
            # print(f'insertion: {insertion_p}')
            # print(f'true_next_base: {labels_1}')
            # print(f'next_base: {next_base}, insertion: {insertion}')
            # print(f'true base: {true_next_base}, true insertion: {true_insertion}')

            total += labels_1.size(0)
            # correct_next_base = count_matching_indices(next_base, true_next_base)
            # correct_insertion = count_matching_indices(insertion, true_insertion)
            for i in range(next_base.shape[0]):
                if next_base[i] == true_next_base[i]:
                    next_base_correct += 1
                    # else:
                    #    print(
                    #        f"input: {utils.decode_kmer(inputs[i], k=config.training.kmer)}, next_base: {utils.CLASSES_PROB_1[next_base[i]]}, true_next_base: {utils.CLASSES_PROB_1[true_next_base[i]]}"
                    #    )
                if insertion[i] == true_insertion[i]:
                    insertion_correct += 1
                if (
                    next_base[i] == true_next_base[i]
                    and insertion[i] == true_insertion[i]
                ):
                    correct += 1
    print(
        f"# of correct next base: {next_base_correct}, # of correct insertion: {insertion_correct}"
    )
    print(f"# of corrent next base & insertion: {correct}, total: {total}")
    val_loss = running_loss / len(test_loader)
    accuracy = 100 * correct / total
    return val_loss, accuracy


def train_only_first(
    model, train_loader, test_loader, criterion, optimizer, writer, config
):
    save_interval = 10
    epochs = config.training.epochs
    model_dir = config.training.model_path + "/" + start_time
    ensure_dir(model_dir)
    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for i, (inputs, labels_1, labels_2) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, labels_1, labels_2 = (
                inputs.to(device),
                labels_1.to(device),
                labels_2.to(device),
            )
            # print(f'num: {i+1}, input shape:{inputs.shape}')
            # print(f'label1 shape: {labels_1.shape}, label2 shape: {labels_2.shape}')
            # print(f'inputs: {inputs}')
            # print(f'label: {labels_1}')
            next_base, insertion = model(inputs)
            # print(f'training next_base: {next_base}, training label: {labels_1}')
            loss = criterion(next_base, labels_1)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        writer.add_scalar("Loss/train", epoch_loss, epoch)

        val_loss, accuracy = test_only_first(
            model,
            test_loader,
            criterion,
        )
        writer.add_scalar("Loss/test", val_loss, epoch)
        writer.add_scalar("Accuracy/test", accuracy, epoch)
        writer.flush()
        print(
            f"Time:{datetime.now()}, Epoch {epoch+1}/{epochs}, Training loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}",
            flush=True,
        )
        print(f"val loss: {val_loss:.4f}")
        if (epoch + 1) % save_interval == 0:
            model_save_path = os.path.join(
                model_dir, f"{config.training.out_predix}_epoch_{epoch+1}.pt"
            )
            torch.save(model.state_dict(), model_save_path)
            model.load_state_dict(torch.load(model_save_path, map_location=device))
            print(f"Model saved at epoch {epoch+1}", flush=True)


def test_only_first(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels_1, labels_2 in test_loader:
            inputs, labels_1, labels_2 = (
                inputs.to(device),
                labels_1.to(device),
                labels_2.to(device),
            )

            next_base_p, insertion_p = model(inputs)
            # loss = criterion(outputs, labels)
            loss_1 = criterion(next_base_p, labels_1)
            loss = loss_1

            running_loss += loss.item()
            _, next_base = torch.max(next_base_p.data, 1)
            _, true_next_base = torch.max(labels_1.data, 1)
            # print(f'next_base: {next_base}')
            # print(f'true base: {true_next_base}')

            total += labels_1.size(0)
            # correct_next_base = count_matching_indices(next_base, true_next_base)
            # correct_insertion = count_matching_indices(insertion, true_insertion)
            for i in range(next_base.shape[0]):
                if next_base[i] == true_next_base[i]:
                    correct += 1
    print(f"# of corrent next base: {correct}, total: {total}")
    val_loss = running_loss / len(test_loader)
    accuracy = 100 * correct / total
    return val_loss, accuracy


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="defaults.yaml",
)
def main(config: DictConfig) -> None:
    os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
    config = config.error_prediction
    print(OmegaConf.to_yaml(config), flush=True)

    # tensorboard
    writer = SummaryWriter(config.data_path.tensorboard_f)
    dataset = Data_Loader(
        file_path=config.data_path.label_f,
        config=config,
        chunk_size=config.training.data_loader_chunk_size,
    )
    train_loader, test_loader = create_data_loader(
        dataset,
        batch_size=config.training.batch_size,
        train_ratio=config.training.train_ratio,
    )

    # update up_seq_len if the input encoder is kmer encoder
    if config.training.encoder == "kmer":
        config.training.up_seq_len = (
            config.training.up_seq_len - config.training.kmer + 1
        )

    # model = Baseline().to(device)
    # model = Baseline_Kmer_In(k=config.training.kmer).to(device)
    if config.training.model == "lstm":
        model = nets.LSTM_simple(
            seq_len=config.training.up_seq_len,
            num_layers=config.training.num_layers,
            num_class1=config.training.num_class_1,
            num_class2=config.training.num_class_2,
        ).to(device)
    elif config.training.model == "encoder_transformer":
        model = nets.Encoder_Transformer(
            embed_size=config.training.embed_size,
            vocab_size=config.training.num_tokens**config.training.kmer
            + config.training.kmer_token_shift,
            with_embedding=config.training.with_embedding,
            num_layers=config.training.num_layers,
            forward_expansion=config.training.forward_expansion,
            seq_len=config.training.up_seq_len + 3,  # include [SEP] and class1 class2
            dropout_rate=config.training.drop_out,
            num_class1=config.training.num_class_1,
            num_class2=config.training.num_class_2,
        ).to(device)
    else:
        model = nets.Encoder_Transformer_NoEmbedding(
            heads=config.training.heads,
            num_layers=config.training.num_layers,
            seq_len=config.training.up_seq_len,
            dropout_rate=config.training.drop_out,
            forward_expansion=config.training.forward_expansion,
            num_class1=config.training.num_class_1,
            num_class2=config.training.num_class_2,
        ).to(device)

    summary(
        model,
        input_size=(config.training.batch_size, config.training.up_seq_len + 3),
        device=device,
        depth=4,
    )

    # tensorboard visualize model
    # dummy_input = torch.rand(config.training.batch_size, config.training.up_seq_len).to(
    #    device
    # )
    # writer.add_graph(model, dummy_input)

    criterion_1 = nn.CrossEntropyLoss()
    criterion_2 = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate, weight_decay=1e-3)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.training.learning_rate, weight_decay=1e-3
    )  # type: ignore
    train(
        model,
        train_loader,
        test_loader,
        criterion_1,
        criterion_2,
        optimizer,
        writer,
        config,
    )
    # train_only_first(model, train_loader, test_loader, criterion_1, optimizer, config)
    writer.close()


if __name__ == "__main__":
    print("------------------------Start Training------------------------")
    torch.set_printoptions(threshold=10_000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    # writer = SummaryWriter(f"./errorprediction/runs/experiment-{start_time}")
    print(device, flush=True)
    main()
