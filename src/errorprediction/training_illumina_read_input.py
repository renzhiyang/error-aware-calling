import os
from turtle import forward
import hydra
import gzip

import numpy as np
import torch.optim as optim

from model import *
from data_loader_training import * 
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
#from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter # type: ignore

        
def ont_hot_encoding(label_array, num_class: int):
    '''
        只对 A,C,G,T,-,[PAD] encoding
        A: [1, 0, 0, 0, 0], ..., [PAD]: [0, 0, 0, 0, 0]
    '''
    output = []
    mapping = np.eye(num_class)
    for value in label_array: # type: ignore
        if value == 0:
            output.append(np.zeros(num_class))
        else:
            output.append(np.eye(num_class)[int(value) - 1])
    output = np.array(output, dtype=np.float32)
    return output


def custon_collate_fn(batch):
    if not batch:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])
    inputs, labels_1, labels_2 = zip(*batch)
    inputs = torch.stack([torch.tensor(input, dtype=torch.float32) for input in inputs])
    labels_1 = torch.stack([torch.tensor(label_1, dtype=torch.float32) for label_1 in labels_1])
    labels_2 = torch.stack([torch.tensor(label_2, dtype=torch.float32) for label_2 in labels_2])
    return inputs, labels_1, labels_2


def create_data_loader(dataset, batch_size: int, train_ratio: float):
    '''
        Create DataLoader object for training 
        dataset = [
            [(input, label1, label2), (input, label1, label2), ...],
            [(input, label1, label2), (input, label1, label2), ...],
            ...
        ]
    '''
    dataset = [item for item in dataset if item is not None]
    flat_data = [sample for sublist in dataset for sample in sublist if sample is not None]
    train_size = int(train_ratio * len(dataset))
    test_size = len(flat_data) - train_size
    train_dataset, test_dataset = random_split(flat_data, [train_size, test_size])  # type: ignore
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custon_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=custon_collate_fn)
    
    return train_loader, test_loader

def count_matching_indices(tensor1, tensor2):
    matching_indices = (tensor1 == tensor2)
    num_matching_indices = matching_indices.sum()
    return num_matching_indices

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def train(model, train_loader, test_loader, criterion, optimizer, config):
    save_interval = 10
    epochs = config.training.epochs
    model_dir = config.training.model_path + '/' + start_time
    ensure_dir(model_dir)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        
        for i, (inputs, labels_1, labels_2) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, labels_1, labels_2 = inputs.to(device), labels_1.to(device), labels_2.to(device)
            #print(f'label1 shape: {labels_1.shape}, label2 shape: {labels_2.shape}')
            next_base, insertion = model(inputs)
            
            loss_1 = criterion(next_base, labels_1)
            loss_2 = criterion(insertion, labels_2)
            loss = loss_1 + loss_2
            loss.backward()
            optimizer.step() 
            running_loss += (loss_1.item() + loss_2.item())
            
        epoch_loss = running_loss / len(train_loader)
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        
        val_loss, accuracy = test(model, test_loader, criterion)
        writer.add_scalar('Loss/test', val_loss, epoch)
        writer.add_scalar('Accuracy/test', accuracy, epoch)
        writer.flush()
        print(f'Time:{datetime.now()}, Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}', flush=True)

        if (epoch + 1) % save_interval == 0:
            model_save_path = os.path.join(model_dir, f'{config.training.out_predix}_epoch_{epoch+1}.pt')
            torch.save(model.state_dict(), model_save_path)
            model.load_state_dict(torch.load(model_save_path, map_location=device))
            print(f'Model saved at epoch {epoch+1}', flush=True)
    

def test(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels_1, labels_2 in test_loader:
            inputs, labels_1, labels_2 = inputs.to(device), labels_1.to(device), labels_2.to(device)
            
            next_base_p, insertion_p = model(inputs)
            #loss = criterion(outputs, labels)
            loss_1 = criterion(next_base_p, labels_1)
            loss_2 = criterion(insertion_p, labels_2)
            loss = loss_1 + loss_2
            
            running_loss += loss.item()
            _, next_base = torch.max(next_base_p.data, 1)
            _, insertion = torch.max(insertion_p.data, 1)
            _, true_next_base = torch.max(labels_1.data, 1)
            _, true_insertion = torch.max(labels_2.data, 1)
            #print(f'next_base: {next_base}, insertion: {insertion}')
            #print(f'true base: {true_next_base}, true insertion: {true_insertion}')
            
            total += labels_1.size(0)
            #correct_next_base = count_matching_indices(next_base, true_next_base)
            #correct_insertion = count_matching_indices(insertion, true_insertion)
            for i in range(next_base.shape[0]):
                if next_base[i] == true_next_base[i] and insertion[i] == true_insertion[i]:
                    correct += 1
    
    val_loss = running_loss / len(test_loader)
    accuracy = 100 * correct / total
    return val_loss, accuracy


@hydra.main(version_base=None, config_path='../../configs/error_prediction', config_name='params.yaml')
def main(config: DictConfig) -> None:
    dataset = Data_Loader(file_path=config.data_path.label_f, 
                          config=config,
                          chunk_size=config.training.data_loader_chunk_size)
    train_loader, test_loader = create_data_loader(dataset, 
                                                   batch_size=config.training.batch_size,
                                                   train_ratio=config.training.train_ratio)
    
    model = ErrorPrediction(embed_size=config.training.embed_size, 
                            heads=config.training.heads, 
                            num_layers=config.training.num_layers,
                            forward_expansion=config.training.forward_expansion, 
                            num_tokens=config.training.num_tokens,
                            num_bases = config.training.num_bases,
                            dropout_rate=config.training.dropout_rate, 
                            max_length=config.training.max_length,
                            output_length=config.training.label_length).to(device)
    
    # output model structure
    onnx_input = torch.ones((40,99)).to(device)
    #print(f'pytorch version: {torch.__version__}', flush=True)
    #torch.onnx.export(model, onnx_input, 'model_2class.onnx',  # type: ignore
    #                  input_names=["input sequence"], output_names=["prediction"])
    # tensorboard visualize model
    writer.add_graph(model, onnx_input)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    train(model, train_loader, test_loader, criterion, optimizer, config)
    writer.close()


if __name__ == '__main__':
    print("------------------------Start Training------------------------")
    torch.set_printoptions(threshold=10_000)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(f'runs/experiment-{start_time}')
    print(device, flush=True)
    main()