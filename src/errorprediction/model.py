from json import encoder
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ErrorPrediction_with_CIGAR(nn.Module):
    def __init__(
        self,
        embed_size,
        heads,
        num_layers,
        forward_expansion,
        num_tokens=8,
        num_bases=5,
        dropout_rate=0.1,
        max_length=250,
        output_length=20,
    ):
        super(ErrorPrediction_with_CIGAR, self).__init__()

        self.embed_size = embed_size
        self.output_length = output_length
        self.num_bases = num_bases
        self.token_embedding = nn.Embedding(num_tokens, embed_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        # self.position_embedding = PositionEncoding(d_model=max_length, max_len=max_length)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    embed_size, heads, forward_expansion, dropout=0.1
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(embed_size * max_length, output_length * num_bases)
        self.fc_out_2 = nn.Linear(embed_size * max_length, 2)
        # print(f'fc_out_2 weight: {self.fc_out_2.weight}, bias: {self.fc_out_2.bias}')
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        for layer in self.layers:
            for p in layer.parameters():
                p.data.uniform_(-initrange, initrange)
        self.token_embedding.weight.data.uniform_(-initrange, initrange)
        self.position_embedding.weight.data.uniform_(-initrange, initrange)
        # print(f'initial positon weight:{self.position_embedding.weight}')
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)
        self.fc_out_2.bias.data.zero_()
        self.fc_out_2.bias.data.uniform_(-initrange, initrange)
        self.activation = nn.GELU()

    def forward(self, x, mask=None):
        x = x.long()
        batch_size, seq_length = x.size()  # batch size: 40, seq_lengh: 256
        # print(f'N: {batch_size}, seq_length:{seq_length}')
        embeddings = self.token_embedding(x)
        positions = (
            torch.arange(0, seq_length)
            .unsqueeze(0)
            .expand(batch_size, seq_length)
            .to(x.device)
        )
        # print(f'positions: {positions.shape}')

        x = self.dropout(
            embeddings + self.position_embedding(positions)
        )  # shape: [40, 256, 128]
        # print(f'x shape: {x.shape}')
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=mask)  # [40, 256, 128] -> [40, 256, 128]
        # print(f'x1 shape: {x.shape}')

        x = x.view(x.size(0), -1)  # out shape: [40, 256, 128] -> [40, 256x128]
        # print(f'x2 shape: {x.shape}')

        outputs = self.fc_out(x)  # out shape: [40, 256x128] -> [40, 100]
        # print(f'output1 shape: {outputs.shape}, {outputs.type}')

        outputs = outputs.view(
            batch_size, self.output_length, self.num_bases
        )  # [40, 100] -> [40, 20, 5]
        # print(f'output2 shape: {outputs.shape}')
        return outputs


class ErrorPrediction_with_CIGAR_onlyType(ErrorPrediction_with_CIGAR):
    def forward(self, x, mask=None):
        x = x.long()
        batch_size, seq_length = x.size()  # batch size: 40, seq_lengh: 256
        # print(f'N: {batch_size}, seq_length:{seq_length}')
        # print(f'nan in input: {torch.isnan(x).sum().item()}')
        embeddings = self.token_embedding(x)
        # print(f'nan in embedding:{torch.isnan(self.token_embedding.weight).sum().item()}')
        # print(f'embdding weight:{self.token_embedding.weight}')
        # print(f'before embedding:{x[0].shape}')
        # print(f'after embedding: {embeddings[0].shape}')
        # print(embeddings.weight)
        positions = (
            torch.arange(0, seq_length)
            .unsqueeze(0)
            .expand(batch_size, seq_length)
            .to(x.device)
        )
        # print(f'positions: {positions.shape}')
        # print(f'position: {self.position_embedding(positions)}')
        # print(positions.weight)
        x = self.dropout(embeddings + self.position_embedding(positions))  # shape:
        # print(f'x1 numbr of nan: {torch.isnan(x).sum().item()}, {x.shape}')
        # print(f'number of nan in mask: {torch.isnan(mask).sum().item()}')
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=mask)  #

        x = x.view(x.size(0), -1)  # out shape:
        # print(f'x2 shape: {x.shape}')
        # print(f'x2 number of nan: {torch.isnan(x).sum().item()}, x2: {x} {x.shape}')
        outputs = self.fc_out_2(x)  # out shape: [40, 256x128] -> [40, 2]
        # print(f'output1 shape: {outputs.shape}, {outputs.type}')
        # print(f'output number of nan: {torch.isnan(outputs).sum().item()}, {outputs}')
        # print(outputs.weight)
        # outputs = outputs.view(batch_size, self.output_length, self.num_bases) # [40, 100] -> [40, 20, 5]
        # print(f'output2 shape: {outputs.shape}')
        # outputs = -nn.functional.log_softmax(outputs, dim=1)
        outputs = nn.functional.log_softmax(outputs, dim=1)
        return outputs


class ErrorPrediction(nn.Module):
    def __init__(
        self,
        embed_size,
        heads=8,
        num_layers=1,
        num_class_1=5,
        num_class_2=25,
        forward_expansion=4,
        num_tokens=6,
        num_bases=5,
        dropout_rate=0.1,
        max_length=250,
        output_length=20,
    ):
        super(ErrorPrediction, self).__init__()
        self.embed_size = embed_size
        self.output_length = output_length
        self.num_bases = num_bases
        self.token_embedding = nn.Embedding(num_tokens, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.position_encoding = PositionEncoding(d_model=6, max_len=max_length)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=heads,
            activation="relu",
            batch_first=True,
            dropout=dropout_rate,
        )
        self.layers = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # self.layers = nn.ModuleList([
        #    nn.TransformerEncoderLayer(d_model=embed_size,
        #                               nhead=heads,
        #                               activation='relu',
        #                               batch_first=True,
        #                               dropout=dropout_rate)
        #    for _ in range(num_layers)
        # ])
        # self.pooling = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        # self.fc = nn.Linear(embed_size * max_length, 256)
        self.classifer_1 = nn.Linear(embed_size * max_length, num_class_1)
        self.classifer_2 = nn.Linear(embed_size * max_length, num_class_2)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        # for later in self.layers:
        #    for p in later.parameters():
        #        p.data.uniform_(-initrange, initrange)
        self.token_embedding.weight.data.uniform_(-initrange, initrange)
        self.position_embedding.weight.data.uniform_(-initrange, initrange)
        self.classifer_1.bias.data.zero_()
        self.classifer_1.weight.data.uniform_(-initrange, initrange)
        self.classifer_2.bias.data.zero_()
        self.classifer_2.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        x = x.long()
        # print(x.shape)
        # batch_size, seq_length, _ = x.size() # one-hot encoding
        batch_size, seq_length = x.size()  # nn.embedding
        # print(f'input shape: {x.shape}') # 40x99
        # embeddings = self.token_embedding(x).to(x.device)

        # print(f'embedding shape: {embeddings.shape}') # 40x99x256
        # positions = torch.arange(0, seq_length).unsqueeze(0).expand(batch_size, seq_length).to(x.device)
        x = self.token_embedding(x)
        x = self.position_encoding(x)
        # print(f'after position: {x.shape}')

        # x = self.dropout(x + self.position_embedding(positions)) # without embedding
        # x = self.dropout(embeddings - self.position_embedding(positions)) # with embedding
        x = self.layers(x)
        # x = x.mean(dim=1)
        # for layer in self.layers:
        #    x = layer(x)
        # print(f'after Transformer:{x[0][0]}')
        # print(f'after Transformer shape: {x.shape}') # 40x99x16
        # x = x.permute(0, 2, 1)
        # x = self.pooling(x)
        x = x.view(x.size(0), -1)
        # x = self.relu(self.fc(x))
        output_1 = self.classifer_1(x)
        output_2 = self.classifer_2(x)
        # 3print(f'first: {output_1[0:2]}, second:{output_2[0:2]}')
        # print(f'out1 shape: {output_1.shape}, out2 shape: {output_2.shape}') # 40x5, 40x25
        return output_1, output_2


class PositionEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        # print(x.shape)
        # print(self.pe[:x.size(0)].shape)
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class LSTM(nn.Module):
    def __init__(
        self,
        num_tokens=6,
        embed_size=8,
        num_class_1=5,
        num_class_2=25,
        hidden_dim=128,
        num_layers=1,
        dropout_rate=0.1,
    ):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(num_tokens, embed_size)
        self.conv1 = nn.Conv1d(
            in_channels=embed_size, out_channels=128, kernel_size=5, padding=2
        )
        self.conv2 = nn.Conv1d(
            in_channels=128, out_channels=128, kernel_size=5, padding=2
        )
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim, num_class_1)
        self.fc2 = nn.Linear(hidden_dim, num_class_2)

    def forward(self, x):
        print(f"input shape: {x.shape}")
        x = x.long()
        x = self.embedding(x)
        # print(f'after embedding: {x.shape}')
        x = x.permute(0, 2, 1)
        # print(f'after permute: {x.shape}')

        x = self.conv1(x)
        x = nn.ReLU()(x)
        # print(f'after conv1: {x.shape}')
        x = self.conv2(x)
        x = nn.ReLU()(x)
        # print(f'after conv2: {x.shape}')
        x = self.pool(x)
        # print(f'after pool: {x.shape}')

        x = x.permute(0, 2, 1)
        # print(f'after permute 2: {x.shape}')
        x, _ = self.lstm(x)
        # print(f'after lstm: {x.shape}')
        x = x[:, -1, :]
        # print(f'after: {x.shape}')
        output_1 = self.fc1(x)
        output_2 = self.fc2(x)

        return output_1, output_2


class conv_model(nn.Module):
    def __init__(self, channels=6, num_class_1=5, num_class_2=25, dropout_rate=0.1):
        super(conv_model, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=channels, out_channels=32, kernel_size=11, padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=11, padding=1
        )
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(64 * 24, 128)
        self.fc2_1 = nn.Linear(128, num_class_1)
        self.fc2_2 = nn.Linear(128, num_class_2)

    def forward(self, x):
        x = x.float()
        x = x.permute(0, 2, 1)
        # print(f'input shap: {x.shape}')
        x = self.conv1(x)
        # print(f'after conv1: {x.shape}')
        x = nn.ReLU()(x)
        x = self.pool(x)
        # print(f'after pool1: {x.shape}')

        x = self.conv2(x)
        # print(f'after conv2: {x.shape}')
        x = nn.ReLU()(x)
        x = self.pool(x)
        # print(f'after pool2: {x.shape}')

        x = x.view(x.size(0), -1)
        # print(f'after view: {x.shape}')
        x = self.fc1(x)
        # print(f'after fc1: {x.shape}')
        x = nn.ReLU()(x)
        x = self.dropout(x)

        output_1 = self.fc2_1(x)
        # print(f'out1: {output_1.shape}')
        output_2 = self.fc2_2(x)
        # print(f'out2: {output_2.shape}')

        return output_1, output_2


def generate_mask(src):
    mask = (torch.triu(torch.ones(src, src)) == 1).transpose(0, 1)
    return mask

