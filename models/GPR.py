import numpy as np
import warnings

class GPRBaseClass:
    def __init__(self, input_len, output_len, input_channels, output_channels):
        from sklearn.gaussian_process import GaussianProcessRegressor as sklearn_GaussianProcessRegressor
        #self.models = [sklearn_GaussianProcessRegressor(kernel=None, alpha=1e-10, optimizer="fmin_l_bfgs_b") for _ in range(output_len*output_channels)]
        self.input_len = input_len
        self.output_len = output_len
        self.input_channels = input_channels
        self.output_channels = output_channels
        if self.output_len>1:
            warnings.warn("""
                            This GPR model is suggested for single-step prediction only.\n
                            For multi-step prediction, it is not recommended,\n
                            because the implementation is to train multiple GPR models for each output variable (scalar),\n
                            and the number of output variables is a product of `output_len` and `output_channels`).\n
                            """)

    def fit(self, X, Y):
        assert type(X)==np.ndarray and X.ndim==3, "Input should be a 3-d numpy array"
        assert type(Y)==np.ndarray and Y.ndim==3, "Target should be a 3-d numpy array"
        assert X.shape[0]==Y.shape[0], "The number of samples in input and output should be the same"
        assert X.shape[2]==self.input_channels, "The number of input channels should align with model parameter"
        assert Y.shape[2]==self.output_channels, "The number of output channels should align with model parameter"
        n_samples = X.shape[0]
        X = X.reshape(n_samples, -1) # flatten the input sequence
        Y = Y.reshape(n_samples, -1) # flatten the output sequence
        for i in range(self.output_len*self.output_channels):
            self.models[i].fit(X, Y[:, i])

    def __call__(self, X):
        # X: (N, input_len, n_vars)
        assert type(X)==np.ndarray and X.ndim==3, "Input should be a 3-d numpy array"
        assert X.shape[2]==self.input_channels, "The number of input channels should align with model parameter"
        n_samples = X.shape[0]
        X = X.reshape(n_samples, -1) # flatten the input sequence for linear regression model
        Y_pred=[]
        for i in range(self.output_len*self.output_channels):
            Y_pred.append(self.models[i].predict(X))
        Y_pred = np.array(Y_pred).T # (N, output_len*output_channels)
        Y_pred = Y_pred.reshape(n_samples, self.output_len, self.output_channels) # convert the output to desired shape
        return Y_pred


class GaussianProcessRegressor(GPRBaseClass):
    def __init__(self, input_len, output_len, input_channels, output_channels, kernel=None, alpha=1e-10, optimizer="fmin_l_bfgs_b"):
        super().__init__(input_len, output_len, input_channels, output_channels)
        from sklearn.gaussian_process import GaussianProcessRegressor as sklearn_GaussianProcessRegressor
        self.models = [sklearn_GaussianProcessRegressor(kernel=kernel, alpha=alpha, optimizer=optimizer) for _ in range(output_len*output_channels)]