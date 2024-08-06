#! /usr/bin/env python3

import os
import h5py
import torch
import argparse
import numpy as np
import src.bayesian as bayesian

from src.errorprediction.models.baseline import Baseline
from torch.utils.data import DataLoader


class TensorData:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data_keys = []

        with h5py.File(self.file_path, "r") as h5f:
            self.data_keys = list(h5f.keys())

    def __len__(self):
        return len(self.data_keys)

    def __getitem__(self, idx):
        with h5py.File(self.file_path, "r") as h5f:
            key = self.data_keys[idx]
            grp = h5f[key]
            # position = np.array(grp['position'])[()].item() # type: ignore
            position = np.array(grp["position"]).item()  # type: ignore
            read_one_hot_tensor = torch.tensor(
                grp["tensor_one_hot"][()], dtype=torch.float32
            )  # type: ignore
        return position, read_one_hot_tensor


def predict(args):
    model_fn = args.model
    tensor_fn = args.tensor_fn

    # check file path
    if not os.path.exists(model_fn):
        print(f"Error: Model file '{model_fn}' does not exist.")
        return
    if not os.path.isfile(tensor_fn):
        print(f"Error: Tensor file '{tensor_fn}' does not exist.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.device("mps"):
        device = torch.device("mps")

    print(f"device: {device}")
    model = Baseline().to(device)
    model.load_state_dict(torch.load(model_fn, map_location=device))
    model.eval()

    dataset = TensorData(tensor_fn)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # type: ignore

    predictions = {}
    count = 0
    for position, tensor_one_hot in dataloader:
        count += 1
        if count >= 1000:
            break
        tensor_one_hot = tensor_one_hot.float().to(device)
        next_base_dis, insertion_dis = model(tensor_one_hot)

        position = position.numpy().item()
        next_base_dis = next_base_dis.cpu().detach().numpy().reshape(-1)
        insertion_dis = insertion_dis.cpu().detach().numpy().reshape(-1)
        if position not in predictions:
            predictions[position] = []
        predictions[position].append([next_base_dis, insertion_dis])

    for position in predictions.keys():
        caller = bayesian.BayesianCaller(predictions[position])
        next_base_dis, insertion_dis = caller[0]
        # print(caller.alleles)
        # print(next_base_dis, insertion_dis)


def main():
    parser = argparse.ArgumentParser(
        description="call variants using fine-turned model"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="the fine-turned model that used for prediction",
        required=True,
    )
    parser.add_argument(
        "--tensor_fn",
        type=str,
        help="tensor file generate by generate_tensor.py, h5py file format",
    )
    args = parser.parse_args()
    predict(args)


if __name__ == "__main__":
    main()
