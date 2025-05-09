import torch
import torch.nn as nn
import torch.nn.functional as F
from .baseclass import TimeSeriesNN


class CNN(TimeSeriesNN):
    """
    (batch_size, input_len, input_channels) -> (batch_size, output_len, output_channels)

    A 3-layer CNN with ReLU activation and max-pooling.
    """
    def __init__(self, input_len, output_len, input_channels, output_channels, *args, **kwargs): # For compatibility, we allow extra arguments here, but be sure they are not used.
        super().__init__(input_len, output_len, input_channels, output_channels)

        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(64*(input_len//2//2//2), output_channels*output_len)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch_size, input_len, input_channels) -> (batch_size, input_channels, input_len)
        x = self.conv1(x) # (batch_size, input_channels, input_len) -> (batch_size, 16, input_len)
        x = nn.functional.relu(x)
        x = self.pool1(x) # (batch_size, 16, input_len) -> (batch_size, 16, input_len/2)
        x = self.conv2(x) # (batch_size, 16, input_len/2) -> (batch_size, 32, input_len/2)
        x = nn.functional.relu(x)
        x = self.pool2(x) # (batch_size, 32, input_len/2) -> (batch_size, 32, input_len/4)
        x = self.conv3(x) # (batch_size, 32, input_len/4) -> (batch_size, 64, input_len/4)
        x = nn.functional.relu(x)
        x = self.pool3(x) # (batch_size, 64, input_len/4 -> (batch_size, 64, input_len/8)
        x = x.view(x.size(0), -1)  # (batch_size, 64, input_len/8) -> (batch_size, 64*input_len/8)
        x = self.fc1(x)  # (batch_size, 64*input_len/8) -> (batch_size, output_channels*output_len)
        x = x.view(-1, self.output_len, self.output_channels) # (batch_size, output_channels*output_len) -> (batch_size, output_len, output_channels)
        return x



class CNNLSTM(TimeSeriesNN):
    """
    (batch_size, input_len, input_channels) -> (batch_size, output_len, output_channels)
    
    CNN feature extraction followed by LSTM temporal processing
    """
    def __init__(self, input_len, output_len, input_channels, output_channels,
                    lstm_hidden_size=128,
                    num_lstm_layers=2, *args, **kwargs):
        super().__init__(input_len, output_len, input_channels, output_channels)
        
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=0.2 if num_lstm_layers > 1 else 0
        )
        reduced_length = input_len // 2 // 2 // 2
        self.fc = nn.Linear(reduced_length * lstm_hidden_size, output_channels * output_len)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, (hn, cn) = self.lstm(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(-1, self.output_len, self.output_channels) # (batch_size, output_channels*output_len) -> (batch_size, output_len, output_channels)
        return x



class TCN(TimeSeriesNN):
    """
    (batch_size, input_len, input_channels) -> (batch_size, output_len, output_channels)

    Temporal Convolutional Network (TCN)

    Model architecture inspired by https://github.com/locuslab/TCN/blob/master/TCN/tcn.py

    Required: input_channels == output_channels
    """
    class TCNBlock(nn.Module):
            r"""
            (batch_size, L, input_channels) -> (batch_size, L, output_channels)
            """
            def __init__(self, input_channels, output_channels, kernel_size, dilation, dropout):
                assert input_channels == output_channels, "The number of input and output channels should be equal"
                super(TCN.TCNBlock, self).__init__()
                self.conv = nn.Conv1d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size, dilation=dilation,
                                      padding='same') # Ensure input length does not change after convolution
                self.activation = nn.ReLU()
                self.dropout = nn.Dropout(p=dropout)
            def forward(self, x): # resnet structure
                res=self.conv(x)
                res=self.activation(res)
                res=self.dropout(res)
                return x+res

    def __init__(self, input_len, output_len, input_channels, output_channels, *args,
                num_blocks=4,
                kernel_size=3,
                hidden_dim=64,
                dropout=0.2,
                **kwargs
                ): # For compatibility, we allow extra arguments here, but be sure they are not used.
        super().__init__(input_len, output_len, input_channels, output_channels)
        self.input_conv = nn.Conv1d(in_channels=input_channels,
                                    out_channels=hidden_dim,
                                    kernel_size=kernel_size,
                                    padding='same')
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = 2 ** i
            self.blocks.append(TCN.TCNBlock(input_channels=hidden_dim,
                                            output_channels=hidden_dim,
                                            kernel_size=kernel_size,
                                            dilation=dilation,
                                            dropout=dropout))
        # set two more layers to shorten the output length, in order to reduce the number of parameters of the last fully connected layer
        self.output_conv1 = nn.Conv1d(in_channels=hidden_dim,
                                        out_channels=output_channels,
                                        kernel_size=kernel_size,
                                        padding='same')
        self.output_pool1 = nn.MaxPool1d(kernel_size=2)
        self.output_conv2 = nn.Conv1d(in_channels=output_channels,
                                        out_channels=output_channels,
                                        kernel_size=kernel_size,
                                        padding='same')
        self.output_pool2 = nn.MaxPool1d(kernel_size=2)
        # map the output of the last convolutional layer to the final desired output
        self.fc=nn.Linear((input_len//2//2)*output_channels,output_len*output_channels)

    def forward(self, x): # x.shape: (batch_size, input_len, input_channels)
        x = x.permute(0, 2, 1) # (batch_size, input_len, input_channels) -> (batch_size, input_channels, input_len)
        x = self.input_conv(x) # (batch_size, input_channels, input_len) -> (batch_size, hidden_dim, input_len)
        for block in self.blocks:
            x = block(x) # (batch_size, hidden_dim, input_len) -> (batch_size, hidden_dim, input_len) # dimension does not change
        x = self.output_conv1(x) # (batch_size, hidden_dim, input_len) -> (batch_size, output_channels, input_len)
        x = self.output_pool1(x) # (batch_size, output_channels, input_len) -> (batch_size, output_channels, input_len/2)
        x = self.output_conv2(x) # (batch_size, output_channels, input_len/2) -> (batch_size, output_channels, input_len/2)
        x = self.output_pool2(x) # (batch_size, output_channels, input_len/2) -> (batch_size, output_channels, input_len/4)
        x = x.view(x.size(0), -1) # (batch_size, output_channels, input_len/4) -> (batch_size, output_channels*(input_len/4))
        x = self.fc(x) # (batch_size, output_channels*(input_len/4)) -> (batch_size, output_len*output_channels)
        x = x.view(-1, self.output_len, self.output_channels) # (batch_size, output_len*output_channels) -> (batch_size, output_len, output_channels)
        return x


if __name__ == '__main__':
    # Test the CNN model
    model = CNNLSTM(input_len=100, output_len=10, input_channels=5, output_channels=4)
    x = torch.randn(32, 100, 10)
    y = model(x)
    print(y.shape) # (1, 1, 1)

