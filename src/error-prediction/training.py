import os
import hydra
import gzip

import numpy as np
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
        self.base_int_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3, '-': 4}
        
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
            
            sequence_around = parts[-1].split(':')[1]
            variant_type = parts[-2].split(':')[1]
            read_base = parts[4].split(':')[1]
            truth_base = parts[3].split(':')[1]
            forward_bases = sequence_around[0:self.config.label.window_size_half]
            behind_bases = sequence_around[-self.config.label.window_size_half:]

            input_seq = forward_bases
            label_seq = truth_base
            if variant_type == "SNV":
                input_seq = input_seq + read_base
            elif variant_type == "Insertion":
                input_seq = input_seq + read_base
                label_seq = label_seq + '-' * len(read_base)
            # deletion时不用改变, 详情参照PPT researchprogress_202404
            
            #output debug
            #print(parts, flush=True)
            #print(variant_type, read_base, label_base, forward_bases, behind_bases, len(sequence_around), flush=True)
            input_array = seq_to_array(input_seq, self.base_int_dict)
            label_array = seq_to_array(label_seq, self.base_int_dict)
            num_classes = 5 # 0, 1, 2, 3, 4 | A, C, G, T, - 
            label_array = np.eye(num_classes)[label_array.astype(int)] # one-hot encoding of label
            #print(input_array, label_array, type(input_array), type(label_array), flush=True)
            return input_array, label_array


def seq_to_array(seq: str, map_dict: dict):
    seq = [float(map_dict[char]) for char in seq]
    array = np.array(seq, dtype=np.float32)
    return array
    


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
                          config=config,
                          chunk_size=config.training.data_loader_chunk_size)
    train_loader, test_loader = create_data_loader(dataset, 
                                                   batch_size=config.training.batch_size,
                                                   train_ratio=config.training.train_ratio)
    
    

if __name__ == '__main__':
    main()