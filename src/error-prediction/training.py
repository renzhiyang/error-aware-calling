import os
from turtle import forward
import hydra
import gzip

import numpy as np
import torch.optim as optim
import data_loader_training as data

from typing import List, Dict
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader

class DataEncoder:
    '''
        Encoder for the intermediate output file produced by label_data.py
    '''
    def __init__(self, file_path, config, chunk_size=1000):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.line_offsets = []
        self.total_lines = 0
        self.config = config
        self.vocab = {'A': 1, 'C': 2, 'G': 3, 'T': 4, '-': 5, '[SEP]': 6, '[CLS]': 7, '[EOS]': 8}
        
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f'The file {file_path} does not exist')
        
        with open(file_path, 'r') as file:
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
        
        with open(self.file_path, 'r') as file:
            file.seek(self.line_offsets[chunk_idx])
            for _ in range(line_idx):
                file.readline()
        
            line = file.readline()
            parts = line.strip().split(' ')
            
            position = parts[2].split(':')[1]
            sequence_around = parts[-1].split(':')[1]
            variant_type = parts[-2].split(':')[1]
            read_base = parts[4].split(':')[1]
            truth_base = parts[3].split(':')[1]
            forward_bases = sequence_around[0:self.config.label.window_size_half]
            behind_bases = sequence_around[-self.config.label.window_size_half:]
            
            if variant_type == "Insertion":
                results = []
                for i in range(len(read_base)):
                    update_forward_bases = forward_bases + (i * '-')
                    input_array = input_tokenization(update_forward_bases, read_base[i],
                                                   self.config.training.input_length, self.vocab)
                    label_array = label_tokenization('-', self.vocab)
                    results.append((input_array, label_array))
                    print(f'pos: {position}, Insertion forward base: {update_forward_bases}, read_base:{read_base[i]},'
                          f'truth_seq: -, len_input: {len(input_array)}\n')
                return results
            else:
                if variant_type == "Deletion":
                    forward_bases = forward_bases[:-1]
                
                input_array = input_tokenization(forward_bases, read_base, 
                                               self.config.training.input_length, self.vocab)
                label_array = np.array(truth_base, dtype=np.float32)
                print(f'pos: {position}, {variant_type} forward base: {forward_bases}, len_input:{len(input_array)}'
                      f'read_base:{read_base}, label:{truth_base}\n')
                return input_array, label_array


def input_tokenization(seq_1: str, seq_2:str, max_length:int, vocab: dict):
    '''
        Encoding inputs for Encoder-only Transformer.
        e.g., INPUT, previous bases: AACCTTTT; current base: T
              ENCODED INPUT: [CLS]AACCTTTT[SEP]T
    '''
    array = [vocab["[CLS]"]] \
            + [vocab[char] for char in seq_1] \
            + [vocab["[SEP]"]] \
            + [vocab[char] for char in seq_2]
    while len(array) < max_length:
        array.append(0)
    array = np.array(array, dtype=np.float32)
    return array


def label_tokenization(seq: str, vocab: dict):
    array = [vocab[char] for char in seq] \
            + vocab["[EOS]"]
    array = np.array(array, dtype=np.float32)
    return array


def create_data_loader(dataset: DataEncoder, batch_size: int, train_ratio: float):
    '''
        Create DataLoader object for training 
    '''
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])  # type: ignore
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, test_loader


@hydra.main(version_base=None, config_path='../../configs/error-prediction', config_name='params.yaml')
def main(config: DictConfig) -> None:
    print("main")
    dataset = DataEncoder(file_path=config.data_path.label_f, 
                          config=config,
                          chunk_size=config.training.data_loader_chunk_size)
    train_loader, test_loader = create_data_loader(dataset, 
                                                   batch_size=config.training.batch_size,
                                                   train_ratio=config.training.train_ratio)
    for batch in enumerate(train_loader):
        print('ok')
    

if __name__ == '__main__':
    main()