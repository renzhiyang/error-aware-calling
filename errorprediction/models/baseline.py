import torch.nn as nn

import errorprediction.models.nets as nets


class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.encoder = nets.ResNet18(num_classes=512)
        self.classifer1 = nn.Linear(512, 5)
        self.classifer2 = nn.Linear(512, 25)

    def forward(self, x):
        batch_size, height, width = x.shape
        x = x.view(batch_size, 1, height, width)
        # print(f'input shape: {x.shape}')
        features = self.encoder(x)

        output_1 = self.classifer1(features)
        output_2 = self.classifer2(features)
        return output_1, output_2


class Baseline_Kmer_In(nn.Module):
    def __init__(self, k=3):
        super(Baseline_Kmer_In, self).__init__()
        self.encoder = nets.ResNet1D_18(num_classes=512)
        self.classifer1 = nn.Linear(512, 5)
        self.classifer2 = nn.Linear(512, 25)

    def forward(self, x):
        # batch_size, height = x.shape
        # x = x.view(batch_size, 1, height)
        # print(f"x shape: {x}, {x.shape}")
        features = self.encoder(x)
        output_1 = self.classifer1(features)
        output_2 = self.classifer2(features)
        return output_1, output_2
