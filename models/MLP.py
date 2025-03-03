import torch
import torch.nn as nn
import torch.nn.functional as F
from .baseclass import TimeSeriesNN

class MLP(TimeSeriesNN):
    r"""
    (batch_size, input_len, input_channels) -> (batch_size, output_len, output_channels)  
    A 3-layer MLP with ReLU activation, max pooling, and dropout.
    """
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
    r"""
    (batch_size, input_len, input_channels) -> (batch_size, output_len, output_channels)  
    A simple TSMixer model with a linear token-mixing layer and channel-mixing layer.

    Reference: arXiv:2303.06053v5, Figure 1; https://github.com/ditschuk/pytorch-tsmixer
    """
    class MLP_time(nn.Module):
        r"""
        (batch_size, input_channels, input_len) -> (batch_size, input_channels, input_len)
        """
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
        r"""
        (batch_size, input_len, input_channels) -> (batch_size, input_len, output_channels)
        """
        def __init__(self, input_channels, output_channels, ff_dim, dropout):
            super().__init__()
            self.fc1 = nn.Linear(input_channels, ff_dim)
            self.fc2 = nn.Linear(ff_dim, output_channels)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=dropout)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.dropout(x)
            return x

    class TimeBatchNorm2d(nn.BatchNorm1d):
        r"""
        Copyed from https://github.com/ditschuk/pytorch-tsmixer/blob/main/torchtsmixer/layers.py

        A batch normalization layer that normalizes over the last two dimensions of a sequence in PyTorch, mimicking Keras behavior.

        This class extends nn.BatchNorm1d to apply batch normalization across time and
        feature dimensions.
        """
        def __init__(self, normalized_shape: tuple[int, int]):
            num_time_steps, num_channels = normalized_shape
            super().__init__(num_channels * num_time_steps)
            self.num_time_steps = num_time_steps
            self.num_channels = num_channels

        def forward(self, x):
            assert x.ndim ==3, "Input must have 3 dimensions (N, input_len, input_channels)"
            x = x.reshape(x.shape[0], -1, 1) # Reshaping input to combine time and feature dimensions for normalization
            x = super().forward(x) # Applying batch normalization
            x = x.reshape(x.shape[0], self.num_time_steps, self.num_channels) # Reshaping back to original dimensions (N, S, C)

            return x

    class MixerLayer(nn.Module):
        def __init__(self, input_len, input_channels, output_channels, ff_dim, dropout):
            super().__init__()
            self.mlp_time = TSMixer.MLP_time(input_len=input_len, dropout=dropout)
            self.mlp_feature = TSMixer.MLP_feature(input_channels=input_channels, output_channels=output_channels, ff_dim=ff_dim, dropout=dropout)
            self.bn1 = TSMixer.TimeBatchNorm2d(normalized_shape=(input_len, input_channels))
            self.bn2 = TSMixer.TimeBatchNorm2d(normalized_shape=(input_len, input_channels))
            
            self.projection = (
                nn.Linear(input_channels, output_channels)
                if input_channels != output_channels
                else nn.Identity()
            )

        def forward(self, x):
            # Time Mixing
            res = self.bn1(x)
            res = res.transpose(1, 2) # (batch_size, input_len, input_channels) -> (batch_size, input_channels, input_len)
            res = self.mlp_time(res) # (batch_size, input_channels, input_len) -> (batch_size, input_channels, input_len)
            res = res.transpose(1, 2) # (batch_size, input_channels, input_len) -> (batch_size, input_len, input_channels)
            x = x + res
            # Feature Mixing
            res = self.bn2(x)
            res = self.mlp_feature(res) # (batch_size, input_len, input_channels) -> (batch_size, input_len, output_channels)
            x = self.projection(x) + res
            return x

    def __init__(self, input_len, output_len, input_channels, output_channels, *args,
                    dropout=0.2,
                    n_mixer_layers=3,
                    ff_dim=64,
                    **kwargs
                    ): # For compatibility, we allow extra arguments here, but be sure they are not used.
        r"""
        :param ff_dim: Dimension of the hidden layer in feature MLP.
        """
        super().__init__(input_len, output_len, input_channels, output_channels)
        
        # For first `n_mixer_layers-1` layers, set output_channels=input_channels; only for the last mixer layer, set output_channels=output_channels.
        input_chs = [input_channels for _ in range(n_mixer_layers)] # number of input channels for each mixer layer
        output_chs = [input_channels for _ in range(n_mixer_layers-1)] + [output_channels] # number of output channels for each mixer layer
        self.mixer_layers=nn.ModuleList([TSMixer.MixerLayer(input_len=input_len,
                                                            input_channels=input_ch,
                                                            output_channels=output_ch,
                                                            ff_dim=ff_dim,
                                                            dropout=dropout,
                                                            ) for input_ch, output_ch in zip(input_chs, output_chs)])
        self.temporal_projection = nn.Linear(input_len, output_len)

    def forward(self, x):
        for layer in self.mixer_layers:
            x = layer(x) # (batch_size, input_len, input_channels) -> (batch_size, input_len, output_channels)
        x = x.transpose(1, 2) # (batch_size, input_len, output_channels) -> (batch_size, output_channels, input_len)
        x = self.temporal_projection(x) # (batch_size, output_channels, input_len) -> (batch_size, output_channels, output_len)
        x = x.transpose(1, 2) # (batch_size, output_channels, output_len) -> (batch_size, output_len, output_channels)
        return x








