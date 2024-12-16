"""
本文件定义了几个用于数据预处理的操作
"""
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class GasData:

    class Utils:

        @staticmethod
        def train_test_split(data, train_ratio=0.8, test_ratio=0.2):
            '''
            Split the dataset into training and testing sets, with the ratio specified by `train_ratio` and `test_ratio`.

            Note: The dataset is split by the number of series in the dataset, not the total length of the time series.
            '''
            assert train_ratio+test_ratio<=1.0, "train_ratio+test_ratio must be <= 1.0"
            assert type(data)==list, "`data` must be a list"
            
            n_series=len(data)
            num_train=int(train_ratio*n_series)
            num_test=n_series-num_train

            # Randomly split the dataset into training and testing sets
            indices=list(range(n_series))
            import random
            random.shuffle(indices)
            train_series_indices=indices[:num_train]
            test_series_indices=indices[num_train:num_train+num_test]
            data_train = [data[i] for i in train_series_indices]
            data_test = [data[i] for i in test_series_indices]

            print("train data indices:", train_series_indices)
            print("test data indices:", test_series_indices)
            print("num_train:", num_train)
            print("num_test:", num_test)

            return (train_series_indices, test_series_indices), (data_train, data_test)

        @staticmethod
        def time_series_slice(data, input_len, output_len,
                                input_indices=None,
                                output_indices=None,
                                overlap=0
                                ):
            r'''

            '''
            def split_2d_np(np_2d, input_len, output_len, overlap=0):
                r'''
                If `overlap` is 0, the input and output sequences are non-overlapping.
                If `overlap` is 1, the last input value is at the same time step as the first output value.
                '''
                assert np_2d.ndim==2, "`np_2d` must be a 2D numpy array"
                n_timesteps, n_vars = np_2d.shape
                X = []
                Y = []
                for i in range(0, n_timesteps-input_len-output_len+1, output_len):
                    X.append(np_2d[i:i+input_len,:])
                    Y.append(np_2d[i+input_len-overlap:i+input_len+output_len-overlap,:])
                X=np.array(X).astype("float32") # X shape: (N, input_len, n_vars)
                Y=np.array(Y).astype("float32") # Y shape: (N, output_len, n_vars)
                return X,Y

            assert type(data)==list, f"`data` should be a list, but got {type(data)}"
            assert all([isinstance(item, np.ndarray) for item in data]), "`data` should be a list of numpy arrays, but got non-numpy array in the list"
            assert all([item.ndim==2 for item in data]), "`data` must be a list of 2D numpy arrays, but got non-2D item in the list"
            assert all([item.shape[1]==data[0].shape[1] for item in data]), "All numpy arrays in `data` must have the same number of columns (features)"
            
            n_timesteps, n_vars = data[0].shape
            input_indices=range(n_vars) if input_indices is None else input_indices
            output_indices=range(n_vars) if output_indices is None else output_indices
            
            X_grouped=[]
            Y_grouped=[]
            for data_i in data: # data_i: numpy array. Shape: (n_timesteps, n_vars)
                X_i, Y_i = split_2d_np(data_i, input_len, output_len, overlap)
                X_i, Y_i = X_i[:,:,input_indices], Y_i[:,:,output_indices] # Only take the specified input and output variables into X and Y
                X_grouped.append(X_i) # X_i shape: (N, input_len, len(input_indices))
                Y_grouped.append(Y_i) # Y_i shape: (N, output_len, len(output_indices))

            return X_grouped, Y_grouped

    def __init__(self, data, input_len, output_len,
                    overlap=0,
                    input_indices=None,
                    output_indices=None,
                    var_names=None,
                    var_units=None,
                    transform_func=None,
                    inverse_transform_func=None
                    ):
        r"""
        :param data: list of numpy arrays of shape: (n_timesteps, n_vars). n_timesteps can be different, but n_vars must be the same.
        :param input_len: the length of each input sequence.
        :param output_len: the length of each output sequence.
        :param input_indices: the indices of the input variables. If None, all variables are used as input variables.
        :param output_indices: the indices of the output variables. If None, all variables are used as output variables.
        """
        assert type(data)==list, "data must be a list"
        assert all([isinstance(item, np.ndarray) for item in data]) and all([item.ndim==2 for item in data]), "data must be a list of 2D numpy arrays"
        assert all([item.shape[1]==data[0].shape[1] for item in data]), "All numpy arrays in data must have the same number of columns (features)"

        self.frozen_data = data # save the original data, which is not changed during preprocessing
        self.data = data
        
        self.input_len = input_len
        self.output_len = output_len
        self.overlap = overlap
        self.n_series = len(data)
        self.n_vars = data[0].shape[1]
        self.n_input_vars = self.n_vars if input_indices is None else len(input_indices)
        self.n_output_vars = self.n_vars if output_indices is None else len(output_indices)

        self.input_indices=list(self.n_vars) if input_indices is None else input_indices
        self.output_indices=list(self.n_vars) if output_indices is None else output_indices

        self.var_names=var_names if var_names is not None else [f"var_{i}" for i in range(self.n_vars)]
        self.var_units=var_units if var_units is not None else [f"unit_{i}" for i in range(self.n_vars)]
        self.input_var_names=[self.var_names[i] for i in self.input_indices]
        self.input_var_units=[self.var_units[i] for i in self.input_indices]
        self.output_var_names=[self.var_names[i] for i in self.output_indices]
        self.output_var_units=[self.var_units[i] for i in self.output_indices]

        self.transform_func = lambda x: x if transform_func is None else transform_func
        self.inverse_transform_func = lambda x: x if transform_func is None else inverse_transform_func

        self.data_train = None # to be set by `self.train_test_split()`
        self.data_test = None # to be set by `self.train_test_split()`
        self.train_indices = None # to be set by `self.train_test_split()`
        self.test_indices = None # to be set by `self.train_test_split()`

        self.X_train_grouped = None # to be set by `self.time_series_slice()`
        self.Y_train_grouped = None # to be set by `self.time_series_slice()`
        self.X_test_grouped = None # to be set by `self.time_series_slice()`
        self.Y_test_grouped = None # to be set by `self.time_series_slice()`
        
        self.var_mean = None # to be set by `self.standardize()`
        self.var_std_dev = None # to be set by `self.standardize()`
        self.input_var_mean = None # to be set by `self.standardize()`
        self.input_var_std_dev = None # to be set by `self.standardize()`
        self.output_var_mean = None # to be set by `self.standardize()`
        self.output_var_std_dev = None # to be set by `self.standardize()`
        
        self.X_train = None # to be set by `self.build_train_test_set()`
        self.Y_train = None # to be set by `self.build_train_test_set()`
        self.X_test = None # to be set by `self.build_train_test_set()`
        self.Y_test = None # to be set by `self.build_train_test_set()`

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]
    
    # --------The methods below should be called in order--------
    # --------If the order is changed, bugs may occur------------

    def transform(self):
        r'''
        Apply a transformation function to each numpy array in the data list.
        '''
        self.data = [self.transform_func(d) for d in self.data]
        return

    def standardize(self):
        count=np.sum([d.shape[0] for d in self.data]) # total timestep count
        var_mean = np.sum([np.sum(d, axis=0) for d in self.data], axis=0)/count
        var_std_dev=np.sqrt(np.sum([np.sum((d - var_mean) ** 2, axis=0) for d in self.data], axis=0)/count)
        transformed_data = [(d-var_mean)/var_std_dev for d in self.data]
        
        # Note that `self.data` is changed!
        self.data = transformed_data

        self.var_mean = var_mean
        self.var_std_dev = var_std_dev
        self.input_var_mean = var_mean[self.input_indices]
        self.input_var_std_dev = var_std_dev[self.input_indices]
        self.output_var_mean = var_mean[self.output_indices]
        self.output_var_std_dev = var_std_dev[self.output_indices]

        np.set_printoptions(precision=4,suppress=True,linewidth=200,floatmode='fixed')
        print("input_var_mean:\n", self.input_var_mean)
        print("input_var_std_dev:\n", self.input_var_std_dev)
        print("output_var_mean:\n", self.output_var_mean)
        print("output_var_std_dev:\n", self.output_var_std_dev)

        return transformed_data, \
            (var_mean, var_std_dev), \
            (self.input_var_mean, self.input_var_std_dev), \
            (self.output_var_mean, self.output_var_std_dev)

    def train_test_split(self, train_ratio=0.8, test_ratio=0.2):
        (self.train_indices, self.test_indices), (self.data_train, self.data_test) \
            = self.Utils.train_test_split(self.data, train_ratio, test_ratio)

    def time_series_slice(self):
        r'''
        Segment the time series data into input and output sequences.
        Need to call `self.train_test_split()` first to split the data into training and testing sets.
        '''
        assert hasattr(self, "data_train") and \
            hasattr(self, "data_test"), \
            "Please call `self.train_test_split()` first to split the data into training and testing sets"
        
        self.X_train_grouped, self.Y_train_grouped = self.Utils.time_series_slice(self.data_train, self.input_len, self.output_len, self.input_indices, self.output_indices, self.overlap)
        self.X_test_grouped, self.Y_test_grouped = self.Utils.time_series_slice(self.data_test, self.input_len, self.output_len, self.input_indices, self.output_indices, self.overlap)
        
        print("len(X_train_grouped):", len(self.X_train_grouped))
        print("len(Y_train_grouped):", len(self.Y_train_grouped))
        print("len(X_test_grouped):", len(self.X_test_grouped))
        print("len(Y_test_grouped):", len(self.Y_test_grouped))

        return (self.X_train_grouped, self.Y_train_grouped), (self.X_test_grouped, self.Y_test_grouped)

    def build_train_test_set(self):
        r'''
        Get the training and testing sets (Organized as numpy arrays without a outer list) for time series data.
        Need to call `self.time_series_slice()` first to split the data into input and output sequences.
        '''
        assert hasattr(self, "X_train_grouped") and \
                hasattr(self, "Y_train_grouped") and \
                hasattr(self, "X_test_grouped") and \
                hasattr(self, "Y_test_grouped"), \
                "Please call `self.time_series_slice()` first to split the data into input and output sequences"
        
        self.X_train=np.concatenate(self.X_train_grouped, axis=0) # shape: (N_train, input_len, len(input_indices))
        self.Y_train=np.concatenate(self.Y_train_grouped, axis=0) # shape: (N_train, output_len, len(output_indices))
        self.X_test=np.concatenate(self.X_test_grouped, axis=0) # shape: (N_test, input_len, len(input_indices))
        self.Y_test=np.concatenate(self.Y_test_grouped, axis=0) # shape: (N_test, output_len, len(output_indices))
        
        print("X_train.shape:", self.X_train.shape)
        print("Y_train.shape:", self.Y_train.shape)
        print("X_test.shape:", self.X_test.shape)
        print("Y_test.shape:", self.Y_test.shape)
        
        return (self.X_train, self.Y_train), (self.X_test, self.Y_test)
    
    # -----------------------------------------
    def standardize_2d_np(self, np_2d, mode="input"):
        r'''
        Use the mean and standard deviation of the whole dataset to standardize a 2D numpy array.
        '''
        assert isinstance(np_2d, np.ndarray) and np_2d.ndim==2, "`np_2d` must be a 2D numpy array"
        assert self.var_mean is not None and self.var_std_dev is not None, \
                "Please call `self.standardize()` first to calculate the mean and standard deviation of the whole dataset"

        if mode=="input":
            assert np_2d.shape[1]==self.n_input_vars,\
                "The number of columns (features) of `np_2d` must be the same as the number of input variables in the dataset"            
            return (np_2d - self.input_var_mean) / self.input_var_std_dev
        elif mode=="output":
            assert np_2d.shape[1]==self.n_output_vars,\
                "The number of columns (features) of `np_2d` must be the same as the number of output variables in the dataset"
            return (np_2d - self.output_var_mean) / self.output_var_std_dev
        else:
            raise ValueError("`mode` must be either 'input' or 'output'")

    def inverse_standardize_2d_np(self, np_2d, mode="output"):
        r'''
        Use the mean and standard deviation of the whole dataset to inverse standardize a 2D numpy array.
        '''
        assert isinstance(np_2d, np.ndarray) and np_2d.ndim==2, "`np_2d` must be a 2D numpy array"
        assert self.var_mean is not None and self.var_std_dev is not None, \
                "Please call `self.standardize()` first to calculate the mean and standard deviation of the whole dataset"

        if mode=="output":
            assert np_2d.shape[1]==self.n_output_vars,\
                "The number of columns (features) of `np_2d` must be the same as the number of output variables in the dataset"
            return np_2d * self.output_var_std_dev + self.output_var_mean
        elif mode=="input":
            assert np_2d.shape[1]==self.n_input_vars,\
                "The number of columns (features) of `np_2d` must be the same as the number of input variables in the dataset"
            return np_2d * self.input_var_std_dev + self.input_var_mean
        else:
            raise ValueError("`mode` must be either 'input' or 'output'")
    
    def inverse_transform_2d_np(self, np_2d):
        r'''
        Apply the inverse transformation function to a 2D numpy array.
        '''
        return self.inverse_transform_func(np_2d)


def get_XY_loaders(X, Y,
                    train_ratio=0.7,
                    val_ratio=0.1,
                    test_ratio=0.2,
                    batch_size=32,
                    verbose=1
                    ):
    '''
    Get data loaders for training, validation, and testing, from dataset in np.ndarray format.
    The proportions of training, validation, and testing sets are 0.7, 0.1, and 0.2, respectively.

    Parameters:
    - X: numpy array. Shape: (num_samples, input_len, input_channels)
    - Y: numpy array. Shape: (num_samples, output_len, output_channels)
    - train_ratio: float. The proportion of training samples.
    - val_ratio: float. The proportion of validation samples.
    - test_ratio: float. The proportion of testing samples.
    - batch_size: int.
    - verbose: int. Whether to print messages. If 1, print messages.
    Return:
    - train_loader, val_loader, test_loader
    '''
    assert type(X)==np.ndarray and type(Y)==np.ndarray, 'X and Y must be numpy arrays.'
    assert X.shape[0]==Y.shape[0], 'X and Y must have the same amount of samples.'

    # Customized dataset class # 自定义数据集类
    class TimeSeriesDataset(Dataset):
        def __init__(self, X, Y):
            self.X = X # shape: (num_samples, input_len, input_channels)
            self.Y = Y # shape: (num_samples, output_len, output_channels)

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]

    # Construct dataset # 构建数据集
    X,Y=X.astype("float32"), Y.astype("float32")
    num_samples=X.shape[0]
    input_len=X.shape[1]
    input_channels=X.shape[2]
    output_len=Y.shape[1]
    output_channels=Y.shape[2]

    assert train_ratio+val_ratio+test_ratio<=1.0

    num_train=int(train_ratio*num_samples)
    num_val=int(val_ratio*num_samples)
    num_test=int(test_ratio*num_samples)
    assert num_train+num_val+num_test<=num_samples

    # Randomly split the dataset into training, validation, and testing sets # 随机划分数据集
    indices=list(range(num_samples))
    import random
    random.shuffle(indices)
    train_indices=indices[:num_train]
    val_indices=indices[num_train:num_train+num_val]
    test_indices=indices[num_train+num_val:num_train+num_val+num_test]

    train_dataset = TimeSeriesDataset(X[train_indices], Y[train_indices])
    val_dataset = TimeSeriesDataset(X[val_indices], Y[val_indices])
    test_dataset = TimeSeriesDataset(X[test_indices], Y[test_indices])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    if num_test>0:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    else:
        test_loader=None

    if verbose==1:
        print(f"Train dataset size: X: ({num_train}, {input_len}, {input_channels}); Y: ({num_train}, {output_len}, {output_channels})")
        print(f"Val dataset size: X: ({num_val}, {input_len}, {input_channels}); Y: ({num_val}, {output_len}, {output_channels})")
        print(f"Test dataset size: X: ({num_test}, {input_len}, {input_channels}); Y: ({num_test}, {output_len}, {output_channels})")

    return train_loader, val_loader, test_loader