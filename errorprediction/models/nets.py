import math
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


class ResidualBlock1D(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None
    ):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet1D(nn.Module):
    def __init__(self, block, layers, in_channels=1, num_classes=10):
        super(ResNet1D, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )

        layers = []
        layers.append(
            block(self.in_channels, out_channels, stride=stride, downsample=downsample)
        )
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # reshape input to (batch_size, 1, features)
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def ResNet1D_18(num_classes=10):
    return ResNet1D(
        ResidualBlock1D, [2, 2, 2, 2], in_channels=1, num_classes=num_classes
    )


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(0)
        # print(f"position input shape: {x.shape}")
        # print(f"positional encoding shape: {self.pe.shape}")
        x = x + self.pe[:, :seq_len]
        return x


class LSTM_simple(nn.Module):
    def __init__(
        self, seq_len, hidden_size=128, num_layers=1, num_class1=5, num_class2=25
    ):
        super(LSTM_simple, self).__init__()
        self.lstm = nn.LSTM(seq_len, hidden_size, num_layers, batch_first=True)

        self.fc1 = nn.Linear(hidden_size, num_class1)
        self.fc2 = nn.Linear(hidden_size, num_class2)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)

        out1 = self.fc1(lstm_out)
        out2 = self.fc2(lstm_out)
        return out1, out2


class Encoder_Transformer(nn.Module):
    def __init__(
        self,
        embed_size=56,
        vocab_size=6,
        heads=6,
        num_layers=2,
        with_embedding=True,
        forward_expansion=1024,
        seq_len=99,
        dropout_rate=0.1,
        num_class1=5,
        num_class2=25,
    ):
        super(Encoder_Transformer, self).__init__()
        self.with_embedding = with_embedding
        self.word_channels = embed_size if self.with_embedding else vocab_size
        self.embedding = nn.Embedding(
            vocab_size, embed_size
        )  # just clear here, if not wth_emebdding, then not be used

        self.pos_encoder = PositionalEncoding(self.word_channels, max_len=seq_len)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.word_channels,
            nhead=heads,
            activation="relu",
            dim_feedforward=forward_expansion,
            batch_first=True,
            dropout=dropout_rate,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers
        )

        self.fc1 = nn.Linear(
            self.word_channels * seq_len, num_class1
        )  # For the 1x5 output
        self.fc2 = nn.Linear(
            self.word_channels * seq_len, num_class2
        )  # For the 1x25 output

    def forward(self, x):
        x = x.long()

        if self.with_embedding:
            batch_size, _ = x.size()
            x = self.embedding(x)
        else:
            batch_size, _, _ = x.size()
        x = self.pos_encoder(x.transpose(0, 1))

        encoded_output = self.transformer_encoder(x.transpose(0, 1))
        encoded_output = encoded_output.contiguous().view(batch_size, -1)

        output_1 = self.fc1(encoded_output)
        output_2 = self.fc2(encoded_output)

        output_1 = torch.softmax(output_1, 1)
        output_2 = torch.softmax(output_2, 1)
        return output_1, output_2


class Encoder_Transformer_NoEmbedding(nn.Module):
    def __init__(
        self,
        heads=6,
        num_layers=2,
        forward_expansion=1024,
        seq_len=99,
        dropout_rate=0.1,
        num_class1=5,
        num_class2=25,
    ):
        super(Encoder_Transformer_NoEmbedding, self).__init__()

        self.pos_encoder = PositionalEncoding(1, max_len=seq_len)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=1,
            nhead=heads,
            dim_feedforward=forward_expansion,
            activation="relu",
            batch_first=True,
            dropout=dropout_rate,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers
        )

        self.fc1 = nn.Linear(1 * seq_len, num_class1)  # For the 1x5 output
        self.fc2 = nn.Linear(1 * seq_len, num_class2)  # For the 1x25 output

    def forward(self, x):
        x = x.long()
        batch_size, _ = x.size()

        x = x.unsqueeze(2)
        x = self.pos_encoder(x.transpose(0, 1))

        encoded_output = self.transformer_encoder(x.transpose(0, 1))
        encoded_output = encoded_output.contiguous().view(batch_size, -1)

        output_1 = self.fc1(encoded_output)
        output_2 = self.fc2(encoded_output)

        return output_1, output_2
