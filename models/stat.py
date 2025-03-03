r"""
未与其它模型对齐格式，暂时没用
"""
import numpy as np

class Identical:
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

class ExponentialMovingAverage:
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

class ARIMA:
    pass
