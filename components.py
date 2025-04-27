"""Component functions of the main function. We break the main function into several components to simplify it."""
import numpy as np
import matplotlib
import pandas as pd
import warnings
import os
import sys
import time
import json
import pickle
matplotlib.use('Agg') # Set the backend to disable figure display window
warnings.filterwarnings("ignore", category=UserWarning) # To filter the warning of disabling plt.show()


def get_dataset(args, logger):
    """
    Get everything needed of the dataset.
    
    Notations:
    - X_train, Y_train, X_test, Y_test:
        - Training/testing inputs and outputs.
        - Shape: (N, seq_len, n_vars).
    - X_train_grouped, X_test_grouped, Y_train_grouped, Y_test_grouped:
        - List of numpy arrays of shape (N_i, seq_len, n_vars).
        - Each list item contains samples from a same series.
        - For example, the original data may have 8 series, each with 500 time steps and 6 variables.
            Assume the input and output length are 10 and 5, respectively, and the overlap is 1; assume 7 series are used for training.
            Then, `X_train_grouped` is a list of 7 numpy arrays items, each of shape (50,10,6).
        - Note that the number of total time steps (N_i*seq_len) in each item may vary, since different series may have different lengths.
            In this case, `X_train_grouped` can be expressed as [(12,50,6), (15,50,6), (34,50,6), (19,50,6), (27,50,6)], for instance.
    - input_var_mean, output_var_mean: 1-d numpy array. Mean of input and output variables.
    - input_var_std_dev, output_var_std_dev: 1-d numpy array. Standard deviation of input and output variables.
    - input_var_indices, output_var_indices: List of int. Column indices of input and output variables.
    - input_var_units, output_var_units: List of strings. Units of input and output variables.

    Returns:
        8-Tuple of
        - (X_train, Y_train)
        - (X_test, Y_test)
        - (X_train_grouped, Y_train_grouped)
        - (X_test_grouped, Y_test_grouped)
        - (input_var_mean, output_var_mean)
        - (input_var_std_dev, output_var_std_dev)
        - (input_var_indices, output_var_indices)
        - (input_var_units, output_var_units)
    """

    # Load data
    from data_provider.data_reading import load_data
    #from data_provider.data_reading2 import load_data

    DATA, var_names, var_units = load_data(args.data_path)

    input_var_units = [var_units[var_names.index(var_name)] for var_name in args.input_var_names]
    output_var_units = [var_units[var_names.index(var_name)] for var_name in args.output_var_names]

    input_var_indices = [var_names.index(var_name) for var_name in args.input_var_names]
    output_var_indices = [var_names.index(var_name) for var_name in args.output_var_names]

    # Data preprocessing
    from data_provider.data_preprocessing import TSDL

    (train_dataset, test_dataset), (train_indices, test_indices) =TSDL.train_test_split(TSDL(DATA), train_ratio=0.8, test_ratio=0.2)
    
    if args.transform:
        from functools import partial
        from utils.transform import transform
        var_idx = var_names.index(args.transform_var_name)
        transform_func = partial(transform, slope=args.transform_slope, l_threshold=args.transform_l_threshold, u_threshold=args.transform_u_threshold, var_idx=var_idx)
        train_dataset=TSDL.apply(train_dataset, transform_func)
        test_dataset=TSDL.apply(test_dataset, transform_func)


    var_mean, var_std_dev = train_dataset.mean(), train_dataset.std()
    input_var_mean, input_var_std_dev = var_mean[input_var_indices], var_std_dev[input_var_indices]
    output_var_mean, output_var_std_dev = var_mean[output_var_indices], var_std_dev[output_var_indices]

    train_dataset_standardized = TSDL.standardize(train_dataset, var_mean, var_std_dev)
    test_dataset_standardized = TSDL.standardize(test_dataset, var_mean, var_std_dev)

    X_train_grouped, Y_train_grouped = TSDL.time_series_slice(train_dataset_standardized, args.input_len, args.output_len,
                                                              overlap=args.overlap,
                                                              pool_size=args.pool_size,
                                                              input_var_indices=input_var_indices,
                                                              output_var_indices=output_var_indices)
    X_test_grouped, Y_test_grouped = TSDL.time_series_slice(test_dataset_standardized, args.input_len, args.output_len,
                                                            overlap=args.overlap,
                                                            pool_size=args.pool_size,
                                                            input_var_indices=input_var_indices,
                                                            output_var_indices=output_var_indices)

    X_train = np.concatenate(X_train_grouped, axis=0) # (N_train, input_len, n_input_vars)
    Y_train = np.concatenate(Y_train_grouped, axis=0) # (N_train, output_len, n_output_vars)
    X_test = np.concatenate(X_test_grouped, axis=0) # (N_test, input_len, n_input_vars)
    Y_test = np.concatenate(Y_test_grouped, axis=0) # (N_test, output_len, n_output_vars)

    logger.info("X_train shape: {}".format(X_train.shape))
    logger.info("Y_train shape: {}".format(Y_train.shape))
    logger.info("X_test shape: {}".format(X_test.shape))
    logger.info("Y_test shape: {}".format(Y_test.shape))
    logger.info("Input variable mean: {}".format(input_var_mean))
    logger.info("Input variable std dev: {}".format(input_var_std_dev))
    logger.info("Output variable mean: {}".format(output_var_mean))
    logger.info("Output variable std dev: {}".format(output_var_std_dev))
    logger.info("Train series indices: {}".format(train_indices))
    logger.info("Test series indices: {}".format(test_indices))
    logger.info("Number of train series: {}".format(len(train_indices)))
    logger.info("Number of test series: {}".format(len(test_indices)))

    return (X_train, Y_train), \
        (X_test, Y_test), \
        (X_train_grouped, Y_train_grouped), \
        (X_test_grouped, Y_test_grouped), \
        (input_var_mean, output_var_mean), \
        (input_var_std_dev, output_var_std_dev), \
        (input_var_indices, output_var_indices), \
        (input_var_units, output_var_units)


def model_building_and_training(args, nargs, X_train, Y_train, logger):
    """Build, train, and save a model."""
    model_name=args.model_name
    model_type=nargs.model_type
    
    # Train model based on the model type
    shape_params = {
        'input_len': args.input_len,
        'output_len': args.output_len,
        'input_channels': len(args.input_var_names),
        'output_channels': len(args.output_var_names),
    }
    
    if model_type=='NN':
        import torch.optim as optim
        import torch.nn as nn
        from models.RNN import RNN, LSTM, GRU
        from models.CNN import CNN, CNNLSTM, TCN
        from models.MLP import MLP
        from models.transformer import Transformer, iTransformer, PatchTST, Reformer, Informer
        from models.Linear import LLinear, DLinear, NLinear
        model_dict={
            'MLP': MLP,
            'CNN': CNN,
            'CNNLSTM': CNNLSTM,
            'TCN': TCN,
            'RNN': RNN,
            'LSTM': LSTM,
            'GRU': GRU,
        }
        optimizer_dict={
            'Adam': optim.Adam,
            'SGD': optim.SGD,
            'Adagrad': optim.Adagrad,
            'Adadelta': optim.Adadelta,
            'RMSprop': optim.RMSprop
        }
        model=model_dict[model_name](**shape_params).to(args.device)
        optimizer = optimizer_dict[args.optimizer_name](model.parameters(), lr=args.learning_rate)
        logger.info('Number of NN model parameters: {}'.format(sum(p.numel() for p in model.parameters())))
    
        from utils.fit_history import FitHistory
        from utils.train import train
        from data_provider.data_preprocessing import get_XY_loaders
        train_loader, val_loader, test_loader = get_XY_loaders(X_train, Y_train,
                                                                train_ratio=0.8,
                                                                val_ratio=0.2,
                                                                test_ratio=0.0,
                                                                batch_size=args.batch_size)
        # train the model
        history=FitHistory()
        history.update(
                    *train(model, train_loader, val_loader, optimizer,
                        loss_func=nn.MSELoss(),
                        metric_func=nn.L1Loss(),
                        num_epochs=args.num_epochs,
                        device=args.device,
                        verbose=1)
                    )
        history.summary()
    
    
    elif model_type=='SVR':
        from models.SVR import SVR, LinearSVR, NuSVR
        model_dict={
        'SVR': SVR(**shape_params, C=args.C, epsilon=args.epsilon, kernel=args.kernel, degree=args.degree, tol=args.tol, max_iter=args.max_iter),
        'LinearSVR': LinearSVR(**shape_params, C=args.C, tol=args.tol, max_iter=args.max_iter),
        'NuSVR': NuSVR(**shape_params, C=args.C, nu=args.nu, kernel=args.kernel, degree=args.degree, tol=args.tol, max_iter=args.max_iter),
        }
        model=model_dict[model_name]
        print("Fitting {} model...".format(model_name))
        model.fit(X_train,Y_train)
    
    
    elif model_type=='LR':
        from models.LR import LinearRegression, Ridge, Lasso, ElasticNet
        model_dict={
            "LR": LinearRegression(**shape_params),
            "Ridge": Ridge(**shape_params,alpha=args.l2_ratio),
            "Lasso": Lasso(**shape_params,alpha=args.l1_ratio),
            "ElasticNet": ElasticNet(**shape_params,
                                        alpha=args.l1_ratio+args.l2_ratio,
                                        l1_ratio=args.l1_ratio/(args.l1_ratio+args.l2_ratio) if args.l1_ratio+args.l2_ratio!=0 else 0
                                        ), # See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html for parameter definition.
        }
        model=model_dict[model_name]
        print("Fitting {} model...".format(model_name))
        model.fit(X_train,Y_train)
    
    
    elif model_type=='Ensemble':
        from models.Ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
        tree_params={
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_split": args.min_samples_split,
            "min_samples_leaf": args.min_samples_leaf,
            "max_features": args.max_features,
        }
        model_dict={
            "RandomForest": RandomForestRegressor(**shape_params, **tree_params),
            "ExtraTrees": ExtraTreesRegressor(**shape_params, **tree_params),
            "AdaBoost": AdaBoostRegressor(**shape_params, n_estimators=args.n_estimators, learning_rate=args.ensemble_learning_rate),
            "GradientBoosting": GradientBoostingRegressor(**shape_params, **tree_params, learning_rate=args.ensemble_learning_rate),
        }
        model=model_dict[model_name]
        print("Fitting {} model...".format(model_name))
        model.fit(X_train,Y_train)
    
    
    elif model_type=='GPR':
        from models.GPR import GaussianProcessRegressor
        model_dict={
            "GPR": GaussianProcessRegressor(**shape_params),
        }
        model=model_dict[model_name]
        print("Fitting {} model...".format(model_name))
        model.fit(X_train,Y_train)

    return model

