import numpy as np


class LinearRegressionBaseClass():
    '''
    (N, input_len, n_vars) -> (N, output_len, n_vars)
    Use linear regression to predict the output sequence.
    The channels are independently predicted.
    '''
    def __init__(self, input_len, output_len, input_channels, output_channels):
        from sklearn.linear_model import LinearRegression as sklearn_LinearRegression
        #self.model=sklearn_LinearRegression()
        self.input_len = input_len
        self.output_len = output_len
        self.input_channels = input_channels
        self.output_channels = output_channels

    def fit(self, X, Y):
        assert type(X)==np.ndarray and X.ndim==3, "Input should be a 3-d numpy array"
        assert type(Y)==np.ndarray and Y.ndim==3, "Target should be a 3-d numpy array"
        assert X.shape[0]==Y.shape[0], "The number of samples in input and output should be the same"
        assert X.shape[2]==self.input_channels, "The number of input channels should align with model parameter"
        assert Y.shape[2]==self.output_channels, "The number of output channels should align with model parameter"
        n_samples = X.shape[0]
        X = X.reshape(n_samples, -1) # flatten the input sequence
        Y = Y.reshape(n_samples, -1) # flatten the output sequence
        self.model.fit(X, Y)

    def __call__(self, X):
        # X: (N, input_len, n_vars)
        assert type(X)==np.ndarray and X.ndim==3, "Input should be a 3-d numpy array"
        assert X.shape[2]==self.input_channels, "The number of input channels should align with model parameter"
        n_samples = X.shape[0]
        X = X.reshape(n_samples, -1) # flatten the input sequence for linear regression model
        Y_pred = self.model.predict(X)
        Y_pred = Y_pred.reshape(n_samples, self.output_len, self.output_channels) # convert the output to desired shape
        return Y_pred


class LinearRegression(LinearRegressionBaseClass):
    def __init__(self, input_len, output_len, input_channels, output_channels):
        super().__init__(input_len, output_len, input_channels, output_channels)
        from sklearn.linear_model import LinearRegression as sklearn_LinearRegression
        self.model = sklearn_LinearRegression()


class Ridge(LinearRegressionBaseClass):
    def __init__(self, input_len, output_len, input_channels, output_channels, alpha=1.0):
        super().__init__(input_len, output_len, input_channels, output_channels)
        from sklearn.linear_model import Ridge as sklearn_Ridge
        self.model = sklearn_Ridge(alpha=alpha)


class Lasso(LinearRegressionBaseClass):
    def __init__(self, input_len, output_len, input_channels, output_channels, alpha=1.0):
        super().__init__(input_len, output_len, input_channels, output_channels)
        from sklearn.linear_model import Lasso as sklearn_Lasso
        self.model = sklearn_Lasso(alpha=alpha)


class ElasticNet(LinearRegressionBaseClass):
    def __init__(self, input_len, output_len, input_channels, output_channels, alpha=1.0, l1_ratio=0.5):
        super().__init__(input_len, output_len, input_channels, output_channels)
        from sklearn.linear_model import ElasticNet as sklearn_ElasticNet
        self.model = sklearn_ElasticNet(alpha=alpha, l1_ratio=l1_ratio)