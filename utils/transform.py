import numpy as np


def transform(x, slope=3.0, l_threshold=25, u_threshold=35, var_idx=0):
    """
    Transform the input data using a piecewise linear function.

    The transformed data is obtained by applying the following formula:
        y = x if x < l_threshold
        y = l_threshold + slope * (x - l_threshold) if l_threshold <= x <= u_threshold
        y = x + (u_threshold - l_threshold) * (slope - 1) if x > u_threshold
    The transform is only applied to the variable specified by var_idx.
    
    Args:
        x (numpy.ndarray): Input data with shape (n_samples, n_features).
        slope (float):
        l_threshold (float): Lower threshold.
        u_threshold (float): Upper threshold.
        var_idx (int): Index of the variable to be transformed.
    """
    assert type(x) == np.ndarray, "Input must be a numpy array"
    assert x.ndim in [1, 2], "Input must be a 1D or 2D array"
    if x.ndim == 2:
        result = np.copy(x)

        mask1 = x[:, var_idx] < l_threshold
        mask2 = (x[:, var_idx] >= l_threshold) & (x[:, var_idx] <= u_threshold)
        mask3 = x[:, var_idx] > u_threshold
    
        result[:, var_idx][mask1] = x[:, var_idx][mask1]
        result[:, var_idx][mask2] = l_threshold + slope * (x[:, var_idx][mask2] - l_threshold)
        result[:, var_idx][mask3] = x[:, var_idx][mask3] + (u_threshold-l_threshold) * (slope - 1)
    
    elif x.ndim == 1:
        result = np.copy(x)

        mask1 = x < l_threshold
        mask2 = (x >= l_threshold) & (x <= u_threshold)
        mask3 = x > u_threshold
    
        result[mask1] = x[mask1]
        result[mask2] = l_threshold + slope * (x[mask2] - l_threshold)
        result[mask3] = x[mask3] + (u_threshold-l_threshold) * (slope - 1)

    return result


def inverse_transform(x, slope=3.0, l_threshold=25, u_threshold=35, var_idx=0):
    assert type(x) == np.ndarray, "Input must be a numpy array"
    assert x.ndim in [1, 2], "Input must be a 1D or 2D array"
    if x.ndim == 2:
        result = np.copy(x)
    
        mask1 = x[:, var_idx] < l_threshold
        mask2 = (x[:, var_idx] >= l_threshold) & (x[:, var_idx] <= l_threshold + slope * (u_threshold - l_threshold))
        mask3 = x[:, var_idx] > l_threshold + slope * (u_threshold - l_threshold)
    
        result[:, var_idx][mask1] = x[:, var_idx][mask1]
        result[:, var_idx][mask2] = l_threshold + (x[:, var_idx][mask2] - l_threshold) / slope
        result[:, var_idx][mask3] = x[:, var_idx][mask3] - (u_threshold-l_threshold) * (slope - 1)
    
    elif x.ndim == 1:
        result = np.copy(x)
    
        mask1 = x < l_threshold + slope * (u_threshold - l_threshold)
        mask2 = (x >= l_threshold) & (x <= l_threshold + slope * (u_threshold - l_threshold))
        mask3 = x > l_threshold + slope * (u_threshold - l_threshold)
    
        result[mask1] = x[mask1]
        result[mask2] = l_threshold + (x[mask2] - l_threshold) / slope
        result[mask3] = x[mask3] - (u_threshold-l_threshold) * (slope - 1)

    return result



if __name__ == '__main__':
    x = np.array([[24, 20],
                  [25, 30],
                  [26, 40],
                  [27, 50],
                  [29, 50],
                  [32, 60],
                  [33, 70],
                  [35, 80],
                  [36, 90],
                  [38, 90],
                  [39, 90],
                  [42, 90],
                  [45, 90],
                  [49, 90],]
                )
    for l in [22,24,28,34]:
        for u in [29,34,36,43]:
           print(x-inverse_transform(transform(x)))