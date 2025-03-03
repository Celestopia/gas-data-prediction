import numpy as np
import random
from torch.utils.data import Dataset, DataLoader


class TSDL:
    r"""
    Time Series with Different Lengths

    Example data structure:
    ```
    data = [
        np.random.rand(20,5),
        np.random.rand(30,5),
        np.random.rand(80,5),
        np.random.rand(120,5)
    ]
    tsdl = TSDL(data) # Time series with different lengths, but all has the same number of variables (4).
    ```
    """
    def __init__(self, data):
        r"""
        :param data: list of numpy arrays of shape: (n_timesteps, n_vars). n_timesteps can be different, but n_vars must be the same.
        """
        TSDL.check_validity(data)
        self.series = data
        self.n_series = len(data)
        self.n_vars = data[0].shape[1]

    def __len__(self):
        return self.n_series

    def __getitem__(self, index):
        return self.series[index]

    @staticmethod
    def check_validity(data):
        r"""
        Check if a list object aligns with TSDL data structure.
        """
        assert type(data)==list, f"Object should be a list, but got {type(data)}"
        assert all([type(item) == np.ndarray for item in data]), "Object should be a list of numpy arrays, but got other types in the list"
        assert all([item.ndim == 2 for item in data]), "Object must be a list of 2D numpy arrays, but got non-2D item in the list"
        assert all([item.shape[1] == data[0].shape[1] for item in data]), "All numpy arrays in object must have the same number of columns (features)"

    @staticmethod
    def get_series(tsdl):
        r"""
        Get the list of time series in the TSDL object.
        """
        return tsdl.series # list of numpy arrays of shape: (n_timesteps, n_vars)

    @staticmethod
    def get_mean(tsdl):
        r"""
        Get the mean of all time series in the TSDL object.
        """
        return np.mean(np.concatenate(tsdl.series, axis=0), axis=0)

    @staticmethod
    def get_std_dev(tsdl):
        r"""
        Get the standard deviation of all time series in the TSDL object.
        """
        return np.std(np.concatenate(tsdl.series, axis=0), axis=0)
    
    @staticmethod
    def standardize(tsdl, mean, std):
        r"""
        Standardize the time series in the TSDL object using the mean and standard deviation provided.
        """
        new_tsdl_list = []
        for i in range(tsdl.n_series):
            new_tsdl_list.append((tsdl[i]-mean)/std)
        return TSDL(new_tsdl_list)

    @staticmethod
    def inverse_standardize(tsdl, mean, std):
        r"""
        Inverse standardize the time series in the TSDL object using the mean and standard deviation provided.
        """
        new_tsdl_list = []
        for i in range(tsdl.n_series):
            new_tsdl_list.append((tsdl[i]*std+mean))
        return TSDL(new_tsdl_list)

    @staticmethod
    def time_series_slice(tsdl, input_len, output_len, overlap=0, input_var_indices=None, output_var_indices=None):
        r"""
        Slice the time series data into input and output sequences suitable for supervised learning.

        :param tsdl: TSDL object.
        :param input_len: length of each input sequence.
        :param output_len: length of each output sequence.
        :param input_var_indices: list of indices of variables to be used as input. If None, all variables are used.
        :param output_var_indices: list of indices of variables to be used as output. If None, all variables are used.
        :param overlap: overlap between consecutive input and output sequences.
        
        - If `overlap` is 0, the input and output sequences are non-overlapping.
        - If `overlap` is 1, the last input value is at the same time step as the first output value.

        :return X_grouped, Y_grouped:

        - `X_grouped`: grouped input sequences, list of numpy arrays of shape: `(N_i, input_len, n_input_vars)`. Each list item corresponds to a complete time series in the original data.
        - `Y_grouped`: grouped output sequences, list of numpy arrays of shape: `(N_i, output_len, n_input_vars)`. Each list item corresponds to a complete time series in the original data.
        """
        assert type(tsdl)==TSDL, "tsdl must be a TSDL object, but got {}".format(type(tsdl))
        data = tsdl.series
        if input_var_indices is None:
            input_var_indices=list(range(data[0].shape[1]))
        if output_var_indices is None:
            output_var_indices=list(range(data[0].shape[1]))
        X_grouped=[]
        Y_grouped=[]
        for data_i in data: # data_i: numpy array. Shape: (n_timesteps, n_vars)
            n_timesteps, n_vars = data_i.shape
            X_i_list, Y_i_list = [], []
            for i in range(0, n_timesteps-input_len-output_len+1, output_len):
                X_i_list.append(data_i[i:i+input_len, input_var_indices])
                Y_i_list.append(data_i[i+input_len-overlap:i+input_len+output_len-overlap, output_var_indices])
            X_i=np.array(X_i_list).astype("float32") # X_i shape: (N_i, input_len, len(input_var_indices))
            Y_i=np.array(Y_i_list).astype("float32") # Y_i shape: (N_i, output_len, len(output_var_indices))
            X_grouped.append(X_i)
            Y_grouped.append(Y_i)
        return X_grouped, Y_grouped

    @staticmethod
    def variable_slice(tsdl, var_indices):
        r"""
        Slice the time series data with specified variable indices.

        :param tsdl: TSDL object.
        :param var_indices: list of indices of variables to be kept.
        :return: tsdl object with selected variables.
        """
        assert type(tsdl)==TSDL, "tsdl must be a TSDL object, but got {}".format(type(tsdl))
        new_data=[]
        for data_i in tsdl.series:
            new_data.append(data_i[:,var_indices])
        return TSDL(new_data)

    @staticmethod
    def train_test_split(tsdl, train_ratio=0.8, test_ratio=0.2):
        r"""
        Randomly split the time series data into training and testing sets with corresponding ratios.

        :param tsdl: TSDL object.
        :param train_ratio: float. The proportion of training samples.
        :param test_ratio: float. The proportion of testing samples.
        :return: (train_tsdl, test_tsdl), (train_series_indices, test_series_indices)
        
        - `train_tsdl`: TSDL object for training set.
        - `test_tsdl`: TSDL object for testing set.
        - `train_series_indices`: list of indices for training set.
        - `test_series_indices`: list of indices for testing set.
        """
        assert type(tsdl)==TSDL, "tsdl must be a TSDL object, but got {}".format(type(tsdl))
        assert train_ratio+test_ratio<=1.0, "train_ratio+test_ratio must be <= 1.0"
        num_test = int(test_ratio*tsdl.n_series)+1 # round up to the nearest integer
        num_train = tsdl.n_series - num_test # round down to the nearest integer
        indices = list(range(tsdl.n_series))
        random.shuffle(indices)
        train_series_indices=indices[:num_train]
        test_series_indices=indices[num_train:num_train+num_test]
        train_tsdl = TSDL([tsdl[i] for i in train_series_indices])
        test_tsdl = TSDL([tsdl[i] for i in test_series_indices])
        return (train_tsdl, test_tsdl), (train_series_indices, test_series_indices)


def get_XY_loaders(X, Y,
                    train_ratio=0.7,
                    val_ratio=0.1,
                    test_ratio=0.2,
                    batch_size=32,
                    verbose=1
                    ):
    r"""
    Get data loaders for training, validation, and testing, from dataset in np.ndarray format.

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
    """
    assert type(X)==np.ndarray and type(Y)==np.ndarray, 'X and Y must be numpy arrays.'
    assert X.shape[0]==Y.shape[0], 'X and Y must have the same amount of samples.'

    # Customized dataset class
    class TimeSeriesDataset(Dataset):
        def __init__(self, X, Y):
            self.X = X # shape: (num_samples, input_len, input_channels)
            self.Y = Y # shape: (num_samples, output_len, output_channels)

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]

    # Construct dataset
    X,Y=X.astype("float32"), Y.astype("float32")
    num_samples=X.shape[0]
    input_len=X.shape[1]
    input_channels=X.shape[2]
    output_len=Y.shape[1]
    output_channels=Y.shape[2]

    assert train_ratio+val_ratio+test_ratio<=1.0, "train_ratio+val_ratio+test_ratio must be less than or equal to 1.0"

    num_train=int(train_ratio*num_samples)
    num_val=int(val_ratio*num_samples)
    num_test=int(test_ratio*num_samples)
    assert num_train+num_val+num_test<=num_samples

    # Randomly split the dataset into training, validation, and testing sets
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
    if num_val > 0:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    else:
        val_loader = None
    if num_test > 0:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    else:
        test_loader = None

    if verbose==1:
        print(f"Train dataset size: X: ({num_train}, {input_len}, {input_channels}); Y: ({num_train}, {output_len}, {output_channels})")
        print(f"Val dataset size: X: ({num_val}, {input_len}, {input_channels}); Y: ({num_val}, {output_len}, {output_channels})")
        print(f"Test dataset size: X: ({num_test}, {input_len}, {input_channels}); Y: ({num_test}, {output_len}, {output_channels})")

    return train_loader, val_loader, test_loader