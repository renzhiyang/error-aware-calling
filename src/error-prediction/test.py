import numpy as np
import hydra
import os

from omegaconf import DictConfig, OmegaConf
from typing import List, Dict
from datetime import datetime

# Numeric labels stored as strings
labels = np.array(['1', '2', '3', '9'])

#hydra.main(version_base=None, config_path='../../configs/', config_name='defaults.yaml')
def main(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))
    config = config.label_data
    print(config.data_path.vcf_f)

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    print(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == '__main__':
    #main()
    path = "/home/yang1031/projects/error-aware-calling/src/error-prediction/model_saved/"
    now = datetime.now()
    model_save = path + '/'+ now.strftime("model-%Y%m%d-%H%M%S/")
    print(model_save)
    ensure_dir(model_save)