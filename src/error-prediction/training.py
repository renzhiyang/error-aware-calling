import os
import hydra
import gzip

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
    def __init__(self, file_path, chunk_size=1000):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.line_offsets = []
        self.total_lines = 0
        
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
        parts = line.strip().split('\t')
        print(parts, flush=True)
        return parts


def create_data_loader(dataset: DataEncoder, batch_size: int, train_ratio: float):
    '''
        Create DataLoader object for training 
    '''
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, test_loader


@hydra.main(version_base=None, config_path='../../configs/error-prediction', config_name='params.yaml')
def main(config: DictConfig) -> None:
    print("main")
    dataset = DataEncoder(file_path=config.data_path.label_f, 
                          chunk_size=config.training.data_loader_chunk_size)
    train_loader, test_loader = create_data_loader(dataset, 
                                                   batch_size=config.training.batch_size,
                                                   train_ratio=config.training.train_ratio)
    for batch in train_loader:
        print("start training", flush=True)
        inputs, labels = batch
    

if __name__ == '__main__':
    main()