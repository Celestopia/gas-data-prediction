import numpy as np
import random
from torch.utils.data import Dataset, DataLoader


class TSDL:
    """
    An Aggregation of Time Series with Different Lengths.
    
    Example Usage:
        ```py
        data = [
            np.random.rand(20,5),
            np.random.rand(30,5),
            np.random.rand(80,5),
            np.random.rand(120,5)
        ]
        tsdl = TSDL(data) # Time series with different lengths, but all has the same number of variables (5).
        ```
    """
    def __init__(self, data):
        """
        Args:
            data (list): list of numpy arrays of shape: (n_timesteps, n_vars). n_timesteps can be different, but n_vars must be the same.
        """
        TSDL.check_validity(data)
        self.series = data
        self.n_series = len(data)
        self.n_vars = data[0].shape[1]

    def __len__(self):
        return self.n_series

    def __getitem__(self, index):
        return self.series[index]
    
    def mean(self):
        """Get the mean of all time series in the TSDL object. Return shape: (n_vars,)."""
        return np.mean(np.concatenate(self.series, axis=0), axis=0) # Shape: (n_vars,)

    def std(self):
        """Get the standard deviation of all time series in the TSDL object. Return shape: (n_vars,)."""
        return np.std(np.concatenate(self.series, axis=0), axis=0) # Shape: (n_vars,)

    @property
    def shape(self):
        """Get the shape of all time series in the TSDL object. Return: list of tuples."""
        return [item.shape for item in self.series]

    def variable_slice(self, var_indices):
        """
        Slice the time series data with specified variable indices.

        Args:
            var_indices (list of int): list of indices of variables to be kept.
        
        Returns:
            TSDL object with selected variables.
        """
        new_data=[]
        for data_i in self.series:
            new_data.append(data_i[:,var_indices])
        return TSDL(new_data)

    @staticmethod
    def check_validity(data):
        """Check if a list object aligns with TSDL data structure."""
        assert type(data)==list, f"Object should be a list, but got {type(data)}"
        assert all([type(item) == np.ndarray for item in data]), "Object should be a list of numpy arrays, but got other types in the list."
        assert all([item.ndim == 2 for item in data]), "Object must be a list of 2D numpy arrays, but got non-2D item in the list."
        assert all([item.shape[1] == data[0].shape[1] for item in data]), "All numpy arrays in object must have the same number of columns (features)"

    @staticmethod
    def standardize(tsdl, mean, std):
        """
        Standardize the time series in the TSDL object using the mean and standard deviation provided.

        Args:
            tsdl (TSDL): TSDL object.
            mean (np.ndarray): Mean of all time series. Shape: (n_vars,).
            std (np.ndarray): Standard deviation of all time series. Shape: (n_vars,).
        
        Returns:
            new_tsdl (TSDL): TSDL object with standardized time series.
        """
        new_tsdl_list = []
        for i in range(tsdl.n_series):
            new_tsdl_list.append((tsdl[i]-mean)/std)
        return TSDL(new_tsdl_list)

    @staticmethod
    def inverse_standardize(tsdl, mean, std):
        """
        Inverse standardize the time series in the TSDL object using the mean and standard deviation provided.

        Args:
            tsdl (TSDL): TSDL object.
            mean (np.ndarray): Mean of all time series. Shape: (n_vars,).
            std (np.ndarray): Standard deviation of all time series. Shape: (n_vars,).
        
        Returns:
            new_tsdl (TSDL): TSDL object with inverse standardized time series.
        """
        new_tsdl_list = []
        for i in range(tsdl.n_series):
            new_tsdl_list.append((tsdl[i]*std+mean))
        return TSDL(new_tsdl_list)

    @staticmethod
    def time_series_slice(tsdl, input_len, output_len, overlap=0, pool_size=1,input_var_indices=None, output_var_indices=None):
        r"""
        Slice the time series data into input and output sequences suitable for supervised learning.

        Args:
            tsdl (TSDL): TSDL object.
            input_len (int): Length of each input sequence.
            output_len (int): Length of each output sequence.
            input_var_indices (list of int): List of indices of variables to be used as input. If None, all variables are used.
            output_var_indices (list of int): List of indices of variables to be used as output. If None, all variables are used.
            overlap (int): Overlap between consecutive input and output sequences.
                - If `overlap` is 0, the input and output sequences are non-overlapping.
                - If `overlap` is 1, the last input value is at the same time step as the first output value.
            pool_size (int): Number of time steps to be averaged (see explanations below).

        Returns:
            out (tuple of list): X_grouped, Y_grouped.
                - `X_grouped`: grouped input sequences, list of numpy arrays of shape: `(N_i, input_len, n_input_vars)`. Each list item corresponds to a complete time series in the original data.
                - `Y_grouped`: grouped output sequences, list of numpy arrays of shape: `(N_i, output_len, n_input_vars)`. Each list item corresponds to a complete time series in the original data.
        
        Explanations:
        - Overlap=0:
                             |<----------input_len-------------->|
            Input sequence:  ├───────────────────────────────────┤
            Target sequence:                                     ├────────────────┤
                                                                 |<--output_len-->|

        - Overlap=1:
                             |<----------input_len-------------->|
            Input sequence:  ├───────────────────────────────────┤
            Target sequence:                                      ├────────────────┤
                                                                  |<--output_len-->|

        - Slicing Logic:
            ├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
            |<----------input_seq_0-------------->|<--output_seq_0-->|
                                |<----------input_seq_1-------------->|<--output_seq_1-->|
                                                    |<----------input_seq_2-------------->|<--output_seq_2-->|
            
            For example, assume input_len=10, output_len=5, overlap=1, pool_size=1. The corresponding time index is given by:
            - input_seq_0: [0,10); output_seq_1: [9,14)
            - input_seq_1: [5,15); output_seq_2: [14,19)
            - input_seq_2: [10,20); output_seq_3: [19,24)
            - ...
            - input_seq_i: [i*output_len,i*output_len+input_len); output_seq_i: [i*output_len+input_len-overlap,i*output_len+input_len-overlap+output_len).
        
        - Pooling Logic:
            The input sequence is divided into blocks of size `pool_size`, and the mean of each block is used as the input value.
            
                             |<----------------------------input_len*pool_size--------------------------->|
            Input sequence:  ├────────────────────────────────────────────────────────────────────────────┤
            Target sequence:                                                                              ├────────────────┤
                                                                                                          |<--output_len-->|
        """
        assert type(tsdl)==TSDL, "tsdl must be a TSDL object, but got {}".format(type(tsdl))
        data = tsdl.series
        if input_var_indices is None:
            input_var_indices=list(range(data[0].shape[1])) # Default to using all variables as input.
        if output_var_indices is None:
            output_var_indices=list(range(data[0].shape[1])) # Default to using all variables as output.
        X_grouped=[]
        Y_grouped=[]

        if pool_size == 1: # Normal case

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
        
        elif pool_size > 1:
        
            def get_slice(data_i, start_idx, input_len, pool_size, var_indices):
                d = data_i[start_idx:start_idx+input_len*pool_size, var_indices] # Shape: (input_len*pool_size, n_vars)
                d = d.reshape((input_len, pool_size, -1)) # Shape: (input_len, pool_size, n_vars)
                d = np.mean(d, axis=1) # Shape: (input_len, n_vars)
                return d # Shape: (input_len, len(var_indices))

            for data_i in data: # data_i: numpy array. Shape: (n_timesteps, n_vars)
                n_timesteps, n_vars = data_i.shape
                X_i_list, Y_i_list = [], []
                for i in range(0, n_timesteps-input_len*pool_size-output_len+1, output_len):
                    X_i_list.append(get_slice(data_i, i, input_len, pool_size, input_var_indices))
                    Y_i_list.append(data_i[i+input_len*pool_size-overlap:i+input_len*pool_size+output_len-overlap, output_var_indices])
                X_i=np.array(X_i_list).astype("float32") # Shape: (N_i, input_len, len(input_var_indices))
                Y_i=np.array(Y_i_list).astype("float32") # Shape: (N_i, output_len, len(output_var_indices))
                X_grouped.append(X_i)
                Y_grouped.append(Y_i)

        return X_grouped, Y_grouped

    @staticmethod
    def train_test_split(tsdl, train_ratio=0.8, test_ratio=0.2):
        """
        Randomly split the time series data into training and testing sets with corresponding ratios.

        Note that the number of training and testing samples are counted by the number of timeseries of the tsdl object, not the total number of time steps.
        Each timeseries will remain complete.
        For example, consider a TSDL object with shape [(100,5), (150,5), (180,5), (240,5)].
        Train-test split may yield `train_tsdl` with shape [(100,5), (150,5), (240,5)] and `test_tsdl` with shape [(180,5)].

        Args:
            tsdl (TSDL): TSDL object.
            train_ratio (float): The proportion of training samples.
            test_ratio (float): The proportion of testing samples.
            
        Returns:
            (train_tsdl, test_tsdl), (train_series_indices, test_series_indices).
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

    @staticmethod
    def transform(tsdl, transform_func):
        """
        Apply a transformation function to each time series in the TSDL object.
        """
        new_tsdl_list = []
        for ts in tsdl.series:
            new_tsdl_list.append(transform_func(ts))
        return TSDL(new_tsdl_list)

    @staticmethod
    def apply(tsdl, func, *args, **kwargs):
        """
        Apply a function to each time series in the TSDL object.

        Args:
            tsdl (TSDL): TSDL object.
            func (function): Any function that takes a 2-d numpy array as the first argument and returns a 2-d numpy array of the same shape.
            *args: Additional arguments to be passed to the function.
            **kwargs: Additional keyword arguments to be passed to the function.
        
        Returns:
            new_tsdl (TSDL): TSDL object with transformed time series.

        Examples:
            ```python
            tsdl = TSDL([np.random.rand(100,5), np.random.rand(200,5), np.random.rand(300,5)])
            def scale(ts, multiplier, bias=1):
                return multiplier*ts+bias
            new_tsdl = TSDL.apply(tsdl, scale, 2.0, bias=0.5)
            ```
        """
        new_tsdl_list = []
        for ts in tsdl.series:
            new_tsdl_list.append(func(ts, *args, **kwargs))
        return TSDL(new_tsdl_list)





def get_XY_loaders(X, Y,
                    train_ratio=0.7,
                    val_ratio=0.1,
                    test_ratio=0.2,
                    batch_size=32,
                    verbose=1
                    ):
    """
    Get data loaders for training, validation, and testing, from dataset in np.ndarray format.

    Args:
        X (np.ndarray): Numpy array of shape (num_samples, *).
        Y (np.ndarray): Numpy array of shape (num_samples, *).
        train_ratio (float): The proportion of training samples.
        val_ratio (float): The proportion of validation samples.
        test_ratio (float): The proportion of testing samples.
        batch_size (int): .
        verbose (int): Whether to print messages. If 1, print messages.

    Returns:
        train_loader, val_loader, test_loader
    """
    assert type(X)==np.ndarray and type(Y)==np.ndarray, 'X and Y must be numpy arrays.'
    assert X.shape[0]==Y.shape[0], 'X and Y must have the same amount of samples.'

    # Customized dataset class
    class TimeSeriesDataset(Dataset):
        def __init__(self, X, Y):
            self.X = X # shape: (num_samples, *)
            self.Y = Y # shape: (num_samples, *)

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]

    # Construct dataset
    X,Y=X.astype("float32"), Y.astype("float32")
    num_samples=X.shape[0]

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
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True) if num_val > 0 else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True) if num_test > 0 else None

    if verbose==1:
        X_shape_str = ', '.join([str(i) for i in X.shape[1:]])
        Y_shape_str = ', '.join([str(i) for i in Y.shape[1:]])
        print(f"Train dataset size: X: ({num_train}, {X_shape_str}); Y: ({num_train}, {Y_shape_str})")
        print(f"Val dataset size: X: ({num_val}, {X_shape_str}); Y: ({num_val}, {Y_shape_str})")
        print(f"Test dataset size: X: ({num_test}, {X_shape_str}); Y: ({num_test}, {Y_shape_str})")

    return train_loader, val_loader, test_loader