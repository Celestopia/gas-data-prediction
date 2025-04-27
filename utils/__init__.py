import random
import numpy as np
import torch



def set_seed(seed):
    """Set the random seeds for reproducibility."""
    random.seed(seed) # Seed for random
    np.random.seed(seed) # Seed for numpy
    torch.manual_seed(seed) # Seed for PyTorch of CPU
    torch.cuda.manual_seed(seed) # Seed for PyTorch CUDA of GPU
    torch.cuda.manual_seed_all(seed) # Seed for PyTorch CUDA of all GPUs
    torch.backends.cudnn.deterministic = True # Deterministic mode for cuDNN


def get_model_type(model_name):
    """Return the type (category) of the model based on its name."""
    if model_name in ['MLP', 'CNN', 'LSTM', 'RNN', 'TCN', 'GRU', 'CNNLSTM']:
        return 'NN'
    elif model_name in ['SVR', 'LinearSVR', 'NuSVR']:
        return 'SVR'
    elif model_name in ['LR', 'Lasso', 'Ridge', 'ElasticNet']:
        return 'LR'
    elif model_name in ['RandomForest', 'ExtraTrees', 'AdaBoost', 'GradientBoosting']:
        return 'Ensemble'
    elif model_name in ['GPR']:
        return 'GPR'
    else:
        raise ValueError(f"{model_name} is an invalid model name.")










