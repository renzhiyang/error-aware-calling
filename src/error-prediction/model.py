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
        #self.position_embedding = PositionEncoding(d_model=max_length, max_len=max_length)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(embed_size, heads, forward_expansion, dropout=0.1)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(embed_size * max_length, output_length * num_bases)
        self.fc_out_2 = nn.Linear(embed_size * max_length, 2)
        #print(f'fc_out_2 weight: {self.fc_out_2.weight}, bias: {self.fc_out_2.bias}')
        self.init_weights()
        
    def init_weights(self) -> None:
        initrange = 0.1
        for layer in self.layers:
            for p in layer.parameters():
                p.data.uniform_(-initrange, initrange)
        self.token_embedding.weight.data.uniform_(-initrange, initrange)
        self.position_embedding.weight.data.uniform_(-initrange, initrange)
        #print(f'initial positon weight:{self.position_embedding.weight}')
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)
        self.fc_out_2.bias.data.zero_()
        self.fc_out_2.bias.data.uniform_(-initrange, initrange)
        self.activation = nn.GELU()
    
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
        #print(f'nan in input: {torch.isnan(x).sum().item()}')
        embeddings = self.token_embedding(x)
        #print(f'nan in embedding:{torch.isnan(self.token_embedding.weight).sum().item()}')
        #print(f'embdding weight:{self.token_embedding.weight}')
        #print(f'before embedding:{x[0].shape}')
        #print(f'after embedding: {embeddings[0].shape}')
        #print(embeddings.weight)
        positions = torch.arange(0, seq_length).unsqueeze(0).expand(batch_size, seq_length).to(x.device)
        #print(f'positions: {positions.shape}')
        #print(f'position: {self.position_embedding(positions)}')
        #print(positions.weight)
        x = self.dropout(embeddings + self.position_embedding(positions)) # shape:
        #print(f'x1 numbr of nan: {torch.isnan(x).sum().item()}, {x.shape}') 
        #print(f'number of nan in mask: {torch.isnan(mask).sum().item()}')
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=mask) # 
        
        x = x.view(x.size(0), -1) # out shape: 
        #print(f'x2 shape: {x.shape}')
        #print(f'x2 number of nan: {torch.isnan(x).sum().item()}, x2: {x} {x.shape}')
        outputs = self.fc_out_2(x) # out shape: [40, 256x128] -> [40, 2]
        #print(f'output1 shape: {outputs.shape}, {outputs.type}')
        #print(f'output number of nan: {torch.isnan(outputs).sum().item()}, {outputs}')
        #print(outputs.weight)
        #outputs = outputs.view(batch_size, self.output_length, self.num_bases) # [40, 100] -> [40, 20, 5]
        #print(f'output2 shape: {outputs.shape}')
        #outputs = -nn.functional.log_softmax(outputs, dim=1)
        outputs = nn.functional.log_softmax(outputs, dim=1)
        return outputs
        

class PositionEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        print(x.shape)
        print(self.pe[:x.size(0)].shape)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

def generate_mask(src):
    mask = (torch.triu(torch.ones(src, src))==1).transpose(0,1)
    return mask