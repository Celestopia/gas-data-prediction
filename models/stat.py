import numpy as np

class Identical():
    '''
    (N, input_len, n_vars) -> (N, output_len, n_vars)
    Use the last time element of the input sequence as the predicted value of the output sequence.
    Used in single step prediction typically. If output_len > 1, the output sequence has the same value in each time step.
    Can serve as a baseline.

    Required: input_channels == output_channels
    '''
    def __init__(self, output_len=1, *args, **kwargs): # For compatibility, we allow extra arguments here, but be sure they are not used.
        self.output_len = output_len

    def __call__(self, x):
        # x: (N, input_len, n_vars)
        assert type(x)==np.ndarray and x.ndim==3, "Input should be a 3-d numpy array"
        x_last_step = x[:, -1, :].reshape(x.shape[0], 1, x.shape[2]) # (N, 1, n_vars)
        output = np.tile(x_last_step, reps=(1, self.output_len, 1)) # (N, output_len, n_vars)
        return output

class ExponentialMovingAverage():
    '''
    (N, input_len, n_vars) -> (N, output_len, n_vars)
    Use the exponential moving average of the input sequence as the predicted value of the output sequence.
    Used in single step prediction typically. If output_len > 1, the output sequence has the same value in each time step.
    The channels are independently predicted.
    Can serve as a baseline.
    '''
    def __init__(self, output_len=1, alpha=None, *args, **kwargs): # For compatibility, we allow extra arguments here, but be sure they are not used.
        self.output_len = output_len
        self.alpha = alpha

    def __call__(self, x):
        # x: (N, input_len, n_vars)
        assert type(x)==np.ndarray and x.ndim==3, "Input should be a 3-d numpy array"
        input_len = x.shape[1]
        if self.alpha is None:
            alpha = 2/(1+input_len)
        else: # If alpha is given, use the specified value.
            alpha=self.alpha
        ema = np.zeros_like(x) # create a tensor with the same shape as the input
        ema[:, 0, :] = x[:, 0, :] # use the first value of the input sequence as the initial value of EMA
        for t in range(1, input_len):
            ema[:, t, :] = alpha * x[:, t, :] + (1 - alpha) * ema[:, t - 1, :]
        last_step_ema = ema[:, -1, :].reshape(x.shape[0], 1, x.shape[2]) # use the last step of EMA as the predicted value of the output sequence
        output = np.tile(last_step_ema, reps=(1, self.output_len, 1)) # (N, output_len, n_vars)
        return output

class ARIMA():
    pass



















class SVR():#不能用，待完善
    '''
    (batch_size, input_len, input_channels) -> (batch_size, output_len, output_channels)
    Support Vector Regression
    For each multivariate time series, flatten it to fit SVR's input requirements.
    '''
    def __init__(self, input_len, output_len, input_channels, output_channels,
                hidden_dim=64,
                kernel='rbf',
                C=1.0,
                epsilon=0.1,
                ):
        try:
            from sklearn.svm import SVR as sklearn_SVR
        except ImportError:
            raise ImportError("sklearn is required for SVR module. Please install it using 'pip install scikit-learn' or 'conda install scikit-learn'")
        super().__init__(input_len, output_len, input_channels, output_channels)
        self.fc1=nn.Linear(input_len*input_channels,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,output_len*output_channels)
        self.svr=sklearn_SVR(kernel=kernel, C=C, epsilon=epsilon)


    def forward(self, x, y=None): # x: (batch_size, input_len, input_channels)
        x = x.view(x.size(0), -1) # (batch_size, input_len, input_channels) -> (batch_size, input_len*input_channels)
        x = self.fc1(x) # (batch_size, input_len*input_channels) -> (batch_size, hidden_dim)
        if y is not None:
            self.svr.fit(np.array(x), np.array(y)) # fit SVR model
        x = torch.Tensor(self.svr.predict(np.array(x))) # (batch_size, hidden_dim) -> (batch_size, hidden_dim)
        x = self.fc2(x) # (batch_size, hidden_dim) -> (batch_size, output_len*output_channels)
        x = x.view(x.shape[0], self.output_len, self.output_channels) # (batch_size, output_len*output_channels) -> (batch_size, output_len, output_channels)
        return x