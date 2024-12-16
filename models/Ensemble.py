import numpy as np
import warnings

class EnsembleBaseClass:
    def __init__(self, input_len, output_len, input_channels, output_channels):
        from sklearn.ensemble import RandomForestRegressor as sklearn_RandomForestRegressor
        #self.models = [sklearn_RandomForestRegressor(n_estimators=100, max_depth=5, random_state=0) for _ in range(output_len*output_channels)]
        self.input_len = input_len
        self.output_len = output_len
        self.input_channels = input_channels
        self.output_channels = output_channels
        if self.output_len>1:
            warnings.warn("""
                            This Ensemble model is suggested for single-step prediction only.\n
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


class RandomForestRegressor(EnsembleBaseClass):
    def __init__(self, input_len, output_len, input_channels, output_channels,
                    n_estimators=100,
                    max_depth=5,
                    random_state=0
                    ):
        super().__init__(input_len, output_len, input_channels, output_channels)
        from sklearn.ensemble import RandomForestRegressor as sklearn_RandomForestRegressor
        self.models = [sklearn_RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
                        for _ in range(output_len*output_channels)]

class AdaBoostRegressor(EnsembleBaseClass):
    def __init__(self, input_len, output_len, input_channels, output_channels,
                    n_estimators=50,
                    learning_rate=1.0,
                    random_state=0
                    ):
        super().__init__(input_len, output_len, input_channels, output_channels)
        from sklearn.ensemble import AdaBoostRegressor as sklearn_AdaBoostRegressor
        self.models = [sklearn_AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
                        for _ in range(output_len*output_channels)]


class BaggingRegressor(EnsembleBaseClass):
    def __init__(self, input_len, output_len, input_channels, output_channels,
                    n_estimators=10,
                    max_samples=1.0,
                    max_features=1.0,
                    bootstrap=True,
                    bootstrap_features=False,
                    random_state=0
                    ):
        super().__init__(input_len, output_len, input_channels, output_channels)
        from sklearn.ensemble import BaggingRegressor as sklearn_BaggingRegressor
        self.models = [sklearn_BaggingRegressor(n_estimators=n_estimators,
                                                max_samples=max_samples,
                                                max_features=max_features,
                                                bootstrap=bootstrap,
                                                bootstrap_features=bootstrap_features,
                                                random_state=random_state)
                        for _ in range(output_len*output_channels)]


class GradientBoostingRegressor(EnsembleBaseClass):
    def __init__(self, input_len, output_len, input_channels, output_channels,
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    min_weight_fraction_leaf=0.0,
                    max_features='sqrt',
                    subsample=1.0,
                    random_state=0,
                    verbose=0,
                    max_leaf_nodes=None,
                    warm_start=False
                    ):
        super().__init__(input_len, output_len, input_channels, output_channels)
        from sklearn.ensemble import GradientBoostingRegressor as sklearn_GradientBoostingRegressor
        self.models = [sklearn_GradientBoostingRegressor(n_estimators=n_estimators,
                                                         learning_rate=learning_rate,
                                                         max_depth=max_depth,
                                                         min_samples_split=min_samples_split,
                                                         min_samples_leaf=min_samples_leaf,
                                                         min_weight_fraction_leaf=min_weight_fraction_leaf,
                                                         max_features=max_features,
                                                         subsample=subsample,
                                                         random_state=random_state,
                                                         verbose=verbose,
                                                         max_leaf_nodes=max_leaf_nodes,
                                                         warm_start=warm_start)
                        for _ in range(output_len*output_channels)]


class VotingRegressor(EnsembleBaseClass):
    def __init__(self, input_len, output_len, input_channels, output_channels,
                    estimators,
                    weights=None,
                    n_jobs=1,
                    verbose=0
                    ):
        super().__init__(input_len, output_len, input_channels, output_channels)
        from sklearn.ensemble import VotingRegressor as sklearn_VotingRegressor
        self.models = [sklearn_VotingRegressor(estimators=estimators,
                                               weights=weights, n_jobs=n_jobs, verbose=verbose)
                        for _ in range(output_len*output_channels)]


class StackingRegressor(EnsembleBaseClass):
    def __init__(self, input_len, output_len, input_channels, output_channels,
                    estimators,
                    final_estimator=None,
                    cv=None,
                    n_jobs=1,
                    verbose=0
                    ):
        super().__init__(input_len, output_len, input_channels, output_channels)
        from sklearn.ensemble import StackingRegressor as sklearn_StackingRegressor
        self.models = [sklearn_StackingRegressor(estimators=estimators,
                                                 final_estimator=final_estimator,
                                                 cv=cv, n_jobs=n_jobs, verbose=verbose)
                        for _ in range(output_len*output_channels)]


class ExtraTreesRegressor(EnsembleBaseClass):
    def __init__(self, input_len, output_len, input_channels, output_channels,
                    n_estimators=100,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    min_weight_fraction_leaf=0.0,
                    max_features='sqrt',
                    random_state=0,
                    n_jobs=1,
                    verbose=0
                    ):
        super().__init__(input_len, output_len, input_channels, output_channels)
        from sklearn.ensemble import ExtraTreesRegressor as sklearn_ExtraTreesRegressor
        self.models = [sklearn_ExtraTreesRegressor(n_estimators=n_estimators,
                                                    max_depth=max_depth,
                                                    min_samples_split=min_samples_split,
                                                    min_samples_leaf=min_samples_leaf,
                                                    min_weight_fraction_leaf=min_weight_fraction_leaf,
                                                    max_features=max_features,
                                                    random_state=random_state,
                                                    n_jobs=n_jobs,
                                                    verbose=verbose)
                        for _ in range(output_len*output_channels)]












