"""
本文件定义了几个用于数据预处理的操作
"""
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class GasData(Dataset):
    def __init__(self, data, input_indices=None, output_indices=None):
        r"""
        :param data: list of numpy arrays of shape: (n_timesteps, n_vars). n_timesteps can be different, but n_vars must be the same.
        """
        self.data = data
        self.input_indices = input_indices
        self.output_indices = output_indices

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.transform:
            sample = self.transform(self.data[idx])
        else:
            sample = self.data[idx]
        return sample, self.target[idx]

    def transform(self):
        pass

    def inverse_transform(self):
        pass
















def time_series_standardization(data, mode="multiple"):
    r"""
    Standardize time series data by removing the mean and scaling to unit variance.

    if mode=="single":
        :param data: 2-d numpy array. shape: (n_timesteps, n_channels)
        :return: transformed_data, 2-d numpy array with identical shape as input data.
        :return: var_mean, 1-d numpy array.
        :return: var_std_dev, 1-d numpy array.
        
    elif mode=="multiple":
        :param data: list of numpy arrays of shape: (n_timesteps, n_channels). n_timesteps can be different, but n_channels must be the same.
        :return: transformed_data, list of 2-d numpy arrays with identical shape as input data.
        :return: var_mean, 1-d numpy array.
        :return: var_std_dev, 1-d numpy array.
    """
    if mode=="single": # If only a single 2-d numpy array needs standardization
        assert type(data)==np.ndarray and data.ndim==2, "data must be a 2-d numpy array"
        var_mean = np.mean(data,axis=0)
        var_std_dev = np.std(data,axis=0)
        transformed_data = (data-var_mean)/var_std_dev
        return transformed_data, var_mean, var_std_dev

    elif mode=="multiple": # If multiple 2-d numpy arrays (packed in a list) need standardization
        assert type(data)==list and all([isinstance(item, np.ndarray) for item in data]) and all([item.ndim==2 for item in data]), "data must be a list of 2D numpy arrays"
        assert all([item.shape[1]==data[0].shape[1] for item in data]), "All numpy arrays in data must have the same number of columns (features)"
        count=np.sum([d.shape[0] for d in data]) # total timestep count
        var_mean = np.sum([np.sum(d, axis=0) for d in data], axis=0)/count
        var_std_dev=np.sqrt(np.sum([np.sum((d - var_mean) ** 2, axis=0) for d in data], axis=0)/count)
        transformed_data = [(d-var_mean)/var_std_dev for d in data]
        return transformed_data, var_mean, var_std_dev


def time_series_split(data, input_len, output_len,input_indices=None,output_indices=None,mode="multiple"):
    r'''
    Split a time series data into input and output sequences.

    if mode=="single":
        np.ndarray -> np.ndarray, np.ndarray
        :param data: numpy array. shape: (n_timesteps, n_channels)
        :return: X, Y. X and Y are numpy arrays.
        
    elif mode=="multiple":
        [np.ndarray, ...] -> np.ndarray, np.ndarray, [np.ndarray, ...], [np.ndarray, ...]
        :param data: list of numpy arrays of shape: (n_timesteps, n_channels). n_timesteps can be different, but n_channels must be the same.
        :return: X, Y, X_grouped, Y_grouped. X and Y are numpy arrays, and X_grouped and Y_grouped are lists of numpy arrays.
    
    :param input_len: the desired length of each input sequence.
    :param output_len: the desired length of each output sequence.
    :param input_indices: the indices of the input variables. If None, all variables are used as input variables.
    :param output_indices: the indices of the output variables. If None, all variables are used as output variables.
    '''

    if mode=="single": # If only a single 2-d numpy array needs partitioning
        assert type(data)==np.ndarray and data.ndim==2, "data must be a 2-d numpy array"
        n_steps,n_vars = data.shape
        input_indices=range(n_vars) if input_indices is None else input_indices
        output_indices=range(n_vars) if output_indices is None else output_indices
        X, Y = [], []
        for i in range(0,n_steps-input_len-output_len+1,output_len):
            X.append(data[i:i+input_len,input_indices])
            Y.append(data[i+input_len:i+input_len+output_len,output_indices])
        X, Y = np.array(X).astype("float32"), np.array(Y).astype("float32")
        return X, Y # (N, input_len, n_vars), (N, output_len, len(output_indices))

    elif mode=="multiple": # If multiple 2-d numpy arrays (packed in a list) need partitioning
        assert type(data)==list and all([isinstance(item, np.ndarray) for item in data]) and all([item.ndim==2 for item in data]), "data must be a list of 2D numpy arrays"
        assert all([item.shape[1]==data[0].shape[1] for item in data]), "All numpy arrays in data must have the same number of columns (features)"
        input_indices=range(n_vars) if input_indices is None else input_indices
        output_indices=range(n_vars) if output_indices is None else output_indices
        X_grouped=[]
        Y_grouped=[]
        for data_i in data: # data_i: numpy array. Shape: (mat_data_len, len(var_names))
            data_i_length=data_i.shape[0] # The (time series) length of the current mat data
            X_i=[]
            Y_i=[]
            for i in range(0, data_i_length-input_len-output_len+1, output_len):
                X_i.append(data_i[i:i+input_len,input_indices])
                Y_i.append(data_i[i+input_len:i+input_len+output_len,output_indices]) # When label_len==0, X_i and Y_i don't intersect, and pred_len==output_len
            X_grouped.append(np.array(X_i).astype("float32")) # X_i shape: (N, input_len, n_vars)
            Y_grouped.append(np.array(Y_i).astype("float32")) # Y_i shape: (N, output_len, n_vars)
        X=[]
        Y=[]
        for X_i in X_grouped:
            for X_ij in X_i:
                X.append(X_ij)
        for Y_i in Y_grouped:
            for Y_ij in Y_i:
                Y.append(Y_ij)
        X=np.array(X).astype("float32") # X shape: (N, input_len, n_vars)
        Y=np.array(Y).astype("float32") # Y shape: (N, output_len, len(output_indices))
        return X,Y,X_grouped,Y_grouped
    else:
        raise ValueError("Invalid `mode` argument. Must be 'single' or 'multiple'.")


def train_test_split(X_grouped, Y_grouped, train_ratio=0.8, test_ratio=0.2):
    '''
    Split a grouped dataset into training, and testing sets.
    '''
    assert type(X_grouped)==list and type(Y_grouped)==list and len(X_grouped)==len(Y_grouped), "X_grouped and Y_grouped must be lists of equal length"
    assert all([isinstance(item, np.ndarray) for item in X_grouped]) and all([isinstance(item, np.ndarray) for item in Y_grouped]), "Elements in X_grouped and Y_grouped must be numpy arrays"
    num_groups=len(X_grouped) # number of groups

    assert train_ratio+test_ratio<=1.0
    num_train=int(train_ratio*num_groups)
    num_test=num_groups-num_train

    # Randomly split the dataset into training and testing sets
    indices=list(range(num_groups))
    import random
    random.shuffle(indices)
    train_indices=indices[:num_train]
    test_indices=indices[num_train:num_train+num_test]

    print("train_indices:", train_indices)
    print("test_indices:", test_indices)
    print("num_train:", num_train)
    print("num_test:", num_test)

    X_train_grouped=[X_grouped[i] for i in train_indices]
    Y_train_grouped=[Y_grouped[i] for i in train_indices]
    X_test_grouped=[X_grouped[i] for i in test_indices]
    Y_test_grouped=[Y_grouped[i] for i in test_indices]

    X_train=np.concatenate(X_train_grouped, axis=0)
    Y_train=np.concatenate(Y_train_grouped, axis=0)
    X_test=np.concatenate(X_test_grouped, axis=0)
    Y_test=np.concatenate(Y_test_grouped, axis=0)

    return (train_indices, test_indices),\
            (X_train, Y_train, X_train_grouped, Y_train_grouped), \
            (X_test, Y_test, X_test_grouped, Y_test_grouped)


def get_XY_loaders(X, Y,
                    batch_size=32,
                    verbose=1
                    ):
    '''
    Get data loaders for training, validation, and testing, from dataset in np.ndarray format.
    The proportions of training, validation, and testing sets are 0.7, 0.1, and 0.2, respectively.

    Parameters:
    - X: numpy array. Shape: (num_samples, input_len, input_channels)
    - Y: numpy array. Shape: (num_samples, output_len, output_channels)
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

    train_ratio=0.7 # The proportion of training samples # 训练集占比
    val_ratio=0.1 # The proportion of validation samples # 验证集占比
    test_ratio=0.2 # The proportion of testing samples # 测试集占比
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
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    if verbose==1:
        print(f"Train dataset size: X: ({num_train}, {input_len}, {input_channels}); Y: ({num_train}, {output_len}, {output_channels})")
        print(f"Val dataset size: X: ({num_val}, {input_len}, {input_channels}); Y: ({num_val}, {output_len}, {output_channels})")
        print(f"Test dataset size: X: ({num_test}, {input_len}, {input_channels}); Y: ({num_test}, {output_len}, {output_channels})")

    return train_loader, val_loader, test_loader