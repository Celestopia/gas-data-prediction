import numpy as np
import warnings

class SVRBaseClass:
    def __init__(self, input_len, output_len, input_channels, output_channels):
        from sklearn.svm import SVR as sklearn_SVR
        #self.models = [sklearn_SVR(kernel=kernel, C=C) for _ in range(output_len*output_channels)]
        self.input_len = input_len
        self.output_len = output_len
        self.input_channels = input_channels
        self.output_channels = output_channels
        if self.output_len>1:
            warnings.warn("""
                            This SVR model is suggested for single-step prediction only.\n
                            For multi-step prediction, it is not recommended,\n
                            because the implementation is to train multiple SVR models for each output variable (scalar),\n
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


class SVR(SVRBaseClass):
    def __init__(self, input_len, output_len, input_channels, output_channels,
                    C=1.0,
                    epsilon=0.1,
                    kernel='rbf', # 'linear', 'poly', 'rbf', 'sigmoid'
                    degree=3,
                    tol=0.001,
                    max_iter=-1,
                    ):
        super().__init__(input_len, output_len, input_channels, output_channels)
        from sklearn.svm import SVR as sklearn_SVR
        self.models = [sklearn_SVR(kernel=kernel, C=C, epsilon=epsilon, degree=degree, tol=tol, max_iter=max_iter) 
                        for _ in range(output_len*output_channels)]

class LinearSVR(SVRBaseClass):
    def __init__(self, input_len, output_len, input_channels, output_channels,
                    C=1.0,
                    epsilon=0.0,
                    tol=0.001,
                    max_iter=-1,
                    loss='epsilon_insensitive',
                    ):
        super().__init__(input_len, output_len, input_channels, output_channels)
        from sklearn.svm import LinearSVR as sklearn_LinearSVR
        self.models = [sklearn_LinearSVR(C=C, epsilon=epsilon, loss=loss, tol=tol, max_iter=max_iter)
                        for _ in range(output_len*output_channels)]

class NuSVR(SVRBaseClass):
    def __init__(self, input_len, output_len, input_channels, output_channels,
                    C=1.0,
                    nu=0.5,
                    kernel='rbf', # 'linear', 'poly', 'rbf', 'sigmoid'
                    degree=3,
                    tol=0.001,
                    max_iter=-1,
                    ):
        super().__init__(input_len, output_len, input_channels, output_channels)
        from sklearn.svm import NuSVR as sklearn_NuSVR
        self.models = [sklearn_NuSVR(kernel=kernel, C=C, nu=nu, degree=degree, tol=tol, max_iter=max_iter)
                        for _ in range(output_len*output_channels)]




