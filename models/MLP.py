import torch
import torch.nn as nn
import torch.nn.functional as F
from .baseclass import TimeSeriesNN

class MLP(TimeSeriesNN):
    '''
    (batch_size, input_len, input_channels) -> (batch_size, output_len, output_channels)
    A 3-layer MLP with ReLU activation, max pooling, and dropout.
    Many parameters
    '''
    def __init__(self, input_len, output_len, input_channels, output_channels, *args,
                dropout=0.2,
                **kwargs
                ): # For compatibility, we allow extra arguments here, but be sure they are not used.
        super().__init__(input_len, output_len, input_channels, output_channels)
        self.fc1 = nn.Linear(input_len*input_channels, 256)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(32, output_channels*output_len)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x):
        x=x.reshape(x.size(0), self.input_len*self.input_channels) # (batch_size, input_len, input_channels) -> (batch_size, input_len*input_channels)
        x=self.fc1(x) # (batch_size, input_len*input_channels) -> (batch_size, 256)
        x=nn.functional.relu(x)
        x=self.pool1(x) # (batch_size, 256) -> (batch_size, 128)
        x=self.dropout1(x)
        x=self.fc2(x) # (batch_size, 128) -> (batch_size, 64)
        x=nn.functional.relu(x)
        x=self.pool2(x) # (batch_size, 64) -> (batch_size, 32)
        x=self.dropout2(x)
        x=self.fc3(x) # (batch_size, 32) -> (batch_size, output_channels*output_len)
        x = x.reshape(-1, self.output_len, self.output_channels) # (batch_size, output_channels*output_len) -> (batch_size, output_len, output_channels)
        return x


class TSMixer(TimeSeriesNN):
    '''
    (batch_size, input_len, input_channels) -> (batch_size, output_len, output_channels)
    A simple TSMixer model with a linear token-mixing layer and channel-mixing layer.

    Reference: arXiv:2303.06053v5, Figure 1
    '''
    class MLP_time(nn.Module):
        def __init__(self, input_len, dropout):
            super().__init__()
            self.fc = nn.Linear(input_len, input_len)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=dropout)

        def forward(self, x):
            x = self.fc(x)
            x = self.relu(x)
            x = self.dropout(x)
            return x

    class MLP_feature(nn.Module):
        def __init__(self, input_channels, dropout):
            super().__init__()
            self.fc1 = nn.Linear(input_channels, input_channels)
            self.fc2 = nn.Linear(input_channels, input_channels)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=dropout)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.dropout(x)
            return x

    class MixerLayer(nn.Module):
        def __init__(self, input_len, input_channels, dropout):
            super().__init__()
            self.mlp_time = TSMixer.MLP_time(input_len=input_len, dropout=dropout)
            self.mlp_feature = TSMixer.MLP_feature(input_channels=input_channels, dropout=dropout)
            self.bn1 = nn.BatchNorm2d(input_len)
            self.bn2 = nn.BatchNorm2d(input_channels)

        def forward(self, x):
            # Time Mixing
            res = self.bn1(x)
            res = res.transpose(1, 2) # (batch_size, input_len, input_channels) -> (batch_size, input_channels, input_len)
            res = self.mlp_time(res) # (batch_size, input_channels, input_len) -> (batch_size, input_channels, input_len)
            res = res.transpose(1, 2) # (batch_size, input_channels, input_len) -> (batch_size, input_len, input_channels)
            x = x + res
            # Feature Mixing
            res = self.bn2(x)
            res = self.mlp_feature(res)
            x = x + res
            return x


    def __init__(self, input_len, output_len, input_channels, output_channels, *args,
                    dropout=0.2,
                    n_mixer_layers=3,
                    **kwargs
                    ): # For compatibility, we allow extra arguments here, but be sure they are not used.
        super().__init__(input_len, output_len, input_channels, output_channels)
        self.mixer_layers=nn.ModuleList([TSMixer.MixerLayer(input_len, input_channels, dropout=dropout) for _ in range(n_mixer_layers)])
        self.temporal_projection = nn.Linear(input_len, output_len)

    def forward(self, x):
        for layer in self.mixer_layers:
            x = layer(x)
        x = self.temporal_projection(x)
        return x








