import torch
import torch.nn as nn
import tqdm

# Base class for time series neural network models
class TimeSeriesNN(nn.Module):
    '''
    Input shape: (batch_size, input_len, input_channels)
    Output shape: (batch_size, output_len, output_channels)
    The overall architecture is direct multistep (DMS), rather than iterated multi-step (IMS). That is, directly predicting T future time steps (T>1).
    Usually input_channels == output_channels, but for generalizability, we set them separately.
    '''
    def __init__(self, input_len, output_len, input_channels, output_channels):
        """
        Specify the input and output shapes.
        """
        super(TimeSeriesNN, self).__init__()
        self.input_len = input_len
        self.input_channels = input_channels
        self.output_len = output_len
        self.output_channels = output_channels

    # placeholder for forward function, to be implemented by subclasses
    def forward(self, x): # x: (batch_size, input_len, input_channels)
        pass
    
    # evaluate the model on a given dataset
    def evaluate(self, data,
                loss_func=nn.functional.mse_loss,
                mode='data_loader',
                device='cpu', # note that the default device is 'cpu'
                verbose=1
                ):
        """
        Evaluate the model on a given dataset.

        :param data: a DataLoader object or a 2-tuple of numpy arrays (inputs, targets)
        :param loss_func: a loss function, default is nn.functional.mse_loss
        :param mode: 'data_loader' or 'numpy', default is 'data_loader'
        :param device: device to run the model, default is 'cpu'
        :param verbose: whether to show progress bar, default is 1
        :return: loss on the given dataset
        """
        if mode == 'data_loader': # If mode is 'data_loader', data should be a DataLoader object, e.g. `torch.utils.data.dataloader.DataLoader`.
            data_loader = data
            self.eval()  # switch to evaluation mode
            total_loss = 0.0

            iterator = tqdm.tqdm(data_loader) if verbose==1 else data_loader
            with torch.no_grad():
                for inputs, targets in iterator: # inputs: (batch_size, input_len, input_channels), targets: (batch_size, output_len, output_channels)
                    inputs, targets = inputs.to(device), targets.to(device) # Transfer data to GPU (if available)
                    outputs = self(inputs)
                    total_loss += loss_func(outputs, targets).item() * inputs.size(0)
            return total_loss / len(data_loader.dataset)
        
        elif mode == 'numpy': # If mode is 'numpy', data should be a 2-tuple of numpy arrays
            '''
            data: (inputs, targets)
            - inputs: (batch_size, input_len, input_channels)
            - targets: (batch_size, output_len, output_channels)
            '''
            assert type(data) == tuple and len(data)==2, 'data should be a tuple of 2 elements'
            inputs, targets = data
            assert inputs.ndim==3, 'inputs should be a 3D numpy array, got shape {}'.format(inputs.shape)
            assert targets.ndim==3, 'targets should be a 3D numpy array, got shape {}'.format(targets.shape)
            assert inputs.shape[0]==targets.shape[0], 'inputs and targets should have the same batch size, got {} and {}'.format(inputs.shape[0], targets.shape[0])
            assert inputs.shape[1:]==(self.input_len, self.input_channels), 'inputs should have shape (batch_size, input_len, input_channels)'
            assert targets.shape[1:]==(self.output_len, self.output_channels), 'targets should have shape (batch_size, output_len, output_channels)'
            inputs, targets = torch.from_numpy(inputs).float().to(device), torch.from_numpy(targets).float().to(device)
            self.eval()  # switch to evaluation mode
            with torch.no_grad():
                outputs = self(inputs)
                result = loss_func(outputs, targets).item()
                return result