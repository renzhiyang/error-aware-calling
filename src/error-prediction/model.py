import math
import torch
import torch.nn as nn

class ErrorPrediction_with_CIGAR(nn.Module):
    def __init__(self, embed_size, heads, num_layers, 
                 forward_expansion, num_tokens=8, num_bases=5, dropout_rate=0.1,
                 max_length=250, output_length=20):
        super(ErrorPrediction_with_CIGAR, self).__init__()

        self.embed_size = embed_size
        self.output_length = output_length
        self.num_bases = num_bases
        self.token_embedding = nn.Embedding(num_tokens, embed_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(embed_size, heads, forward_expansion, dropout=0.1)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout_rate)
        print(embed_size * max_length, output_length * num_bases)
        self.fc_out = nn.Linear(embed_size * max_length, output_length * num_bases)
        self.fc_out_2 = nn.Linear(embed_size * max_length, 2)
        
    def forward(self, x, mask=None):
        x = x.long()
        batch_size, seq_length = x.size() # batch size: 40, seq_lengh: 256
        #print(f'N: {batch_size}, seq_length:{seq_length}')
        embeddings = self.token_embedding(x)
        positions = torch.arange(0, seq_length).unsqueeze(0).expand(batch_size, seq_length).to(x.device)
        #print(f'positions: {positions.shape}')
        
        x = self.dropout(embeddings + self.position_embedding(positions)) # shape: [40, 256, 128]
        #print(f'x shape: {x.shape}')
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=mask) # [40, 256, 128] -> [40, 256, 128]
        #print(f'x1 shape: {x.shape}')
        
        x = x.view(x.size(0), -1) # out shape: [40, 256, 128] -> [40, 256x128]
        #print(f'x2 shape: {x.shape}')
        
        outputs = self.fc_out(x) # out shape: [40, 256x128] -> [40, 100]
        #print(f'output1 shape: {outputs.shape}, {outputs.type}')
        
        outputs = outputs.view(batch_size, self.output_length, self.num_bases) # [40, 100] -> [40, 20, 5]
        #print(f'output2 shape: {outputs.shape}')
        return outputs

class ErrorPrediction_with_CIGAR_onlyType(ErrorPrediction_with_CIGAR):
    def forward(self, x, mask=None):
        x = x.long()
        batch_size, seq_length = x.size() # batch size: 40, seq_lengh: 256
        #print(f'N: {batch_size}, seq_length:{seq_length}')
        embeddings = self.token_embedding(x)
        positions = torch.arange(0, seq_length).unsqueeze(0).expand(batch_size, seq_length).to(x.device)
        #print(f'positions: {positions.shape}')
        
        x = self.dropout(embeddings + self.position_embedding(positions)) # shape: [40, 256, 128]
        #print(f'x shape: {x.shape}')
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=mask) # [40, 256, 128] -> [40, 256, 128]
        #print(f'x1 shape: {x.shape}')
        
        x = x.view(x.size(0), -1) # out shape: [40, 256, 128] -> [40, 256x128]
        #print(f'x2 shape: {x.shape}')
        
        outputs = self.fc_out_2(x) # out shape: [40, 256x128] -> [40, 2]
        #print(f'output1 shape: {outputs.shape}, {outputs.type}')
        
        #outputs = outputs.view(batch_size, self.output_length, self.num_bases) # [40, 100] -> [40, 20, 5]
        #print(f'output2 shape: {outputs.shape}')
        return outputs
        

class PositionEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 250):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        if d_model % 2 != 0:
            d_model += 1
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe = torch.zeros(max_len, 1, d_model)
        self.pe[:, 0, 0::2] = torch.sin(position * div_term)
        self.pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', self.pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :, x.size(2)]
        return self.dropout(x)

def generate_mask(src):
    mask = (torch.triu(torch.ones(src, src))==1).transpose(0,1)
    return mask