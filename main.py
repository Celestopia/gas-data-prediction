import numpy as np
import matplotlib
import pandas as pd
import warnings
import torch
import os
import sys
import time
import random
from types import SimpleNamespace


sys.path.append(os.path.join(os.path.dirname(os.path.abspath('__file__')), '..')) # Modify the working path so that this.ipynb file can import other modules like in the root directory
matplotlib.use('Agg') # Set the backend to disable figure display window
warnings.filterwarnings("ignore", category=UserWarning) # To filter the warning of disabling plt.show()


def main(args):
    # Set another argument namespace to store intermediate variables (variables that are not directly input, but can be calculated from the input)
    nargs = SimpleNamespace() # New arguments; To be updated later

    # Set the random seeds
    random.seed(args.seed) # Seed for random
    np.random.seed(args.seed) # Seed for numpy
    torch.manual_seed(args.seed) # Seed for PyTorch of CPU
    torch.cuda.manual_seed(args.seed) # Seed for PyTorch CUDA of GPU
    torch.cuda.manual_seed_all(args.seed) # Seed for PyTorch CUDA of all GPUs
    torch.backends.cudnn.deterministic = True # Deterministic mode for cuDNN

    # Set up the directory to save the results and logs
    result_directory=args.result_directory
    time_string=str(int(time.time()))
    subdirectory='experiment_{}_{}'.format(
                                    time_string,
                                    args.model_name) # Use current timestamp to name subdirectory

    if args.save_result and not os.path.exists(os.path.join(result_directory, subdirectory)):
        os.makedirs(os.path.join(result_directory, subdirectory))
        print("Result directory created at: {}".format(os.path.join(result_directory, subdirectory)))

    nargs.time_string = time_string
    nargs.subdirectory = subdirectory

    # Set the logger
    from debug.logger import get_logger # Import personalized logger
    log_path = os.path.join(result_directory, subdirectory, 'log.log') # Set the log path
    logger = get_logger(log_file_path=log_path) # Get the logger

    # Determine model type
    if args.model_name in ['MLP', 'CNN', 'LSTM', 'RNN', 'TCN', 'GRU']:
        nargs.model_type='NN'
    elif args.model_name in ['SVR', 'LinearSVR', 'NuSVR']:
        nargs.model_type='SVR'
    elif args.model_name in ['LR', 'Lasso', 'Ridge', 'ElasticNet']:
        nargs.model_type='LR'
    elif args.model_name in ['RandomForest', 'ExtraTrees', 'AdaBoost', 'GradientBoosting']:
        nargs.model_type='Ensemble'
    elif args.model_name in ['GPR']:
        nargs.model_type='GPR'
    else:
        raise ValueError(f"{args.model_name} is an invalid model name.")

    print("----------------Model Setting Information----------------")
    logger.info("Model name: {}".format(args.model_name))
    logger.info("Input variables: {}".format(args.input_var_names))
    logger.info("Output variables: {}".format(args.output_var_names))
    logger.info("Input length: {}".format(args.input_len))
    logger.info("Output length: {}".format(args.output_len))
    logger.info("Input channels: {}".format(len(args.input_var_names)))
    logger.info("Output channels: {}".format(len(args.output_var_names)))
    logger.info("Overlap: {}".format(args.overlap))
    logger.info("Result directory: {}".format(args.result_directory))
    logger.info("Seed: {}".format(args.seed))
    print("---------------------------------------------------------")

    from components import get_dataset, model_building_and_training, model_evaluation, save_plots, save_result, save_objects

    (X_train, Y_train), \
        (X_test, Y_test), \
        (X_train_grouped, Y_train_grouped), \
        (X_test_grouped, Y_test_grouped), \
        (input_var_mean, output_var_mean), \
        (input_var_std_dev, output_var_std_dev), \
        (input_var_indices, output_var_indices), \
        (input_var_units, output_var_units) = get_dataset(args, logger)

    nargs.input_var_units = input_var_units
    nargs.input_var_mean = input_var_mean
    nargs.input_var_std_dev = input_var_std_dev
    nargs.output_var_units = output_var_units
    nargs.output_var_mean = output_var_mean
    nargs.output_var_std_dev = output_var_std_dev

    model= model_building_and_training(args, nargs, X_train, Y_train, logger)
    
    from utils.model_test import ModelTest
    Exp=ModelTest(model=model, device=args.device) # Experiment object initialization
    
    prediction_info, Y_true, Y_pred = model_evaluation(args, nargs, Exp, X_test_grouped, Y_test_grouped, logger)

    if args.save_plots is True:
        save_plots(args, nargs, Exp, Y_pred, Y_true)
    save_result(args, nargs, prediction_info)
    if args.save_objects is True:
        save_objects(args, nargs, model, Y_true, Y_pred)
    return prediction_info

# Default parameter dictionary
param_dict = {
    "model_name": "CNN",
    "data_path": r'E:\科创优才\实验数据\GasDataset\DataMining\data1628.pkl',
    "seed": 1234,
    
    # Dataset Settings
    "input_var_names": [
        "Environment Temperature",
        "Pumping Flow",
        "Inside Temperature",
        "Inside Humidity",
        "Inside Pressure",
        "Outside Temperature",
        "Outside Humidity",
        "Outside Pressure",
        "FGR",
        "Fan Frequency",
        "Load",
        "Gas Bias Valve",
        "Fan Bias Valve",
        "Flame Temperature",
        "Flame Speed",
    ], # The names of input variables.
    "output_var_names": [
        #"O2",
        #"CO2",
        "NOx",
        "Smoke Temperature",
    ], # The names of output variables.
    "input_len": 30, # The temporal length of input sequence.
    "output_len": 1, # The temporal length of output sequence.
    "overlap": 1, # The overlap between input and output sequences.

    # Saving Settings
    "save_result": True, # Whether to save the result files.
    "result_directory": "./results", # The directory to save all results. The results of each single run will be saved in a sub-directory under this directory.
    "figsize": (18,12), # The size of the figure.
    "save_log": True, # Whether to save the running log.
    "save_plots": True, # Whether to save the plots.
    "save_objects": True, # Whether to save the python objects (model, intermediate data, etc.).
    
    # Model Settings

    ## Linear Regression Settings
    "l2_ratio": 0.1, # The scale parameter of L2 regularization.
    "l1_ratio": 0.1, # The scale parameter of L1 regularization.
    
    ## SVR Settings
    "C": 1.0, # The parameter C of SVR.
    "epsilon": 0.1, # The parameter epsilon of Linear SVR.
    "kernel": 'rbf', # The kernel function of SVR. options: ["rbf", "linear", "poly", "sigmoid"].
    "degree": 3, # Degree of the polynomial kernel function ("poly"). Must be non-negative. Ignored by all other kernels.
    "nu": 0.5, # The parameter nu of NuSVR.
    "tol": 1e-3, # The tolerance of the stopping criterion.
    "max_iter": -1, # The maximum number of iterations. -1 means no limit.

    ## Ensemble Settings
    "n_estimators": 100, # The number of trees in the forest.
    "max_depth": 5, # The maximum depth of the tree.
    "min_samples_split": 2, # The minimum number of samples required to split an internal node.
    "min_samples_leaf": 1, # The minimum number of samples required to be at a leaf node.
    "max_features":'sqrt', # The number of features to consider when looking for the best split. Options: ["auto", "sqrt", "log2"].
    "ensemble_learning_rate": 0.1, # The learning rate of the ensemble model. Used for gradient boosting.

    ## Neural Network Settings
    "device": "cpu", # device to train the neural network model. Options: ["cuda", "cpu"].
    "optimizer_name": "Adam", # The optimizer used for training. Options: ["Adam", "SGD", "Adagrad", "Adadelta", "RMSprop"].
    "learning_rate": 0.001, # learning rate
    "batch_size": 32, # batch size
    "num_epochs": 100, # maximum number of epochs (may stop earlier due to early stopping settings)
}



if __name__ == '__main__':
    t0=time.time() # Start the timer
    args = SimpleNamespace(**param_dict) # Convert the dictionary to a namespace object for easier access to the parameters.
    main(args)

