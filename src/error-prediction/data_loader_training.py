import os
import hydra
import gzip

from omegaconf import DictConfig, OmegaConf
from typing import List, Dict

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