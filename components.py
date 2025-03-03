r"""
Component functions of the main function. We break the main function into several components to simplify it.
"""
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
    r"""
    Get everything needed of the dataset.
    """

    # Load data
    from data_provider.data_reading import load_data

    DATA, var_names, var_units = load_data(args.data_path)

    input_var_units = [var_units[var_names.index(var_name)] for var_name in args.input_var_names]
    output_var_units = [var_units[var_names.index(var_name)] for var_name in args.output_var_names]

    input_var_indices = [var_names.index(var_name) for var_name in args.input_var_names]
    output_var_indices = [var_names.index(var_name) for var_name in args.output_var_names]

    # Data preprocessing
    from data_provider.data_preprocessing import TSDL

    (train_dataset, test_dataset), (train_indices, test_indices) =TSDL.train_test_split(TSDL(DATA), train_ratio=0.8, test_ratio=0.2)

    var_mean, var_std_dev = TSDL.get_mean(train_dataset), TSDL.get_std_dev(train_dataset)
    input_var_mean, input_var_std_dev = var_mean[input_var_indices], var_std_dev[input_var_indices]
    output_var_mean, output_var_std_dev = var_mean[output_var_indices], var_std_dev[output_var_indices]

    train_dataset_standardized = TSDL.standardize(train_dataset, var_mean, var_std_dev)
    test_dataset_standardized = TSDL.standardize(test_dataset, var_mean, var_std_dev)

    X_train_grouped, Y_train_grouped = TSDL.time_series_slice(train_dataset_standardized, args.input_len, args.output_len,
                                                              overlap=args.overlap,
                                                              input_var_indices=input_var_indices,
                                                              output_var_indices=output_var_indices)
    X_test_grouped, Y_test_grouped = TSDL.time_series_slice(test_dataset_standardized, args.input_len, args.output_len,
                                                            overlap=args.overlap,
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
    r"""
    Build, train, and save a model.
    """
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
        from models.CNN import CNN, TCN
        from models.MLP import MLP
        from models.transformer import Transformer, iTransformer, PatchTST, Reformer, Informer
        from models.Linear import LLinear, DLinear, NLinear
        model_dict={
            'MLP': MLP,
            'CNN': CNN,
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

def model_evaluation(args, nargs, Exp, X_test_grouped, Y_test_grouped, logger):
    with_Tensor=True if nargs.model_type=='NN' else False
    Y_pred, Y_true = Exp.get_pred_true_series_pairs(X_test_grouped[0], Y_test_grouped[0], with_Tensor=with_Tensor) # visualize on the first sample of test set
    prediction_info=Exp.get_prediction_info(X_test_grouped, Y_test_grouped,
                                            args.output_var_names, nargs.output_var_units, nargs.output_var_mean, nargs.output_var_std_dev,
                                            with_Tensor=with_Tensor)
    return prediction_info, Y_true, Y_pred

def save_plots(args, nargs, Exp, Y_pred, Y_true):
    r"""
    Save four plots:
    - pred_std: standardized prediction
    - pred: rescaled prediction
    - res_std: standardized residual
    - res: rescaled residual
    """
    for plot_name, (p,r) in zip(["pred_std","pred","res_std","res"], [(0,0), (0,1), (1,0), (1,1)]):
        Exp.plot_all_predictions(Y_pred, Y_true,
                                output_var_names=args.output_var_names,
                                output_var_units=nargs.output_var_units,
                                output_var_mean=nargs.output_var_mean,
                                output_var_std_dev=nargs.output_var_std_dev,
                                plot_residual=p,
                                rescale=r,
                                figsize=args.figsize,
                                save_path='{}/{}/{}_{}.png'.format(
                                    args.result_directory,
                                    nargs.subdirectory,
                                    args.model_name,
                                    plot_name))
    return

def save_result(args, nargs, prediction_info):
    r"""
    Save model experiment information to result.json
    """
    hyperparameter_dict = {}

    if nargs.model_type=='NN':
        hyperparameter_dict = {
            "optimizer_name": args.optimizer_name,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
        }
    elif nargs.model_type=='SVR':
        hyperparameter_dict = {
            "C": args.C,
            "epsilon": args.epsilon,
            "nu": args.nu,
            "kernel": args.kernel,
            "degree": args.degree,
            "tol": args.tol,
            "max_iter": args.max_iter,
        }
    elif nargs.model_type=='LR':
        hyperparameter_dict = {
            "l1_ratio": args.l1_ratio,
            "l2_ratio": args.l2_ratio,
        }
    elif nargs.model_type=='Ensemble':
        hyperparameter_dict = {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_split": args.min_samples_split,
            "min_samples_leaf": args.min_samples_leaf,
            "max_features": args.max_features,
            "ensemble_learning_rate": args.ensemble_learning_rate,
        }
    elif nargs.model_type=='GPR':
        hyperparameter_dict = {
            "default": "default",
        }

    result_dict = {
        "time_string": nargs.time_string,
        "model_name": args.model_name,
        "dataset_info": {
            "input_len": args.input_len,
            "output_len": args.output_len,
            "input_channels": len(args.input_var_names),
            "output_channels": len(args.output_var_names),
            "overlap": args.overlap,
            "input_var_names": args.input_var_names,
            "output_var_names": args.output_var_names,
        },
        "prediction_info": prediction_info,
        "hyperparameters": hyperparameter_dict,
        "seed": args.seed,
    }

    save_path = os.path.join(args.result_directory, nargs.subdirectory, 'result.json')
    with open(save_path, 'w') as f:
        json.dump(result_dict, f, indent=4)
    return

def save_objects(args, nargs, model, Y_true, Y_pred):
    """
    Save the model and experiment objects as a .pkl file, in order that they can be loaded later for further analysis or visualization
    """
    objects_pkl_save_path = os.path.join(args.result_directory, nargs.subdirectory, 'objects.pkl')
    objects_pkl_dict = {
        "model": model,
        "Y_pred": Y_pred,
        "Y_true": Y_true,

        "input_var_names": args.input_var_names,
        "input_var_units": nargs.input_var_units,
        "input_var_mean": nargs.input_var_mean,
        "input_var_std_dev": nargs.input_var_std_dev,

        "output_var_names": args.output_var_names,
        "output_var_mean": nargs.output_var_mean,
        "output_var_std_dev": nargs.output_var_std_dev,
        "output_var_units": nargs.output_var_units,
        
        "args": args,
        "nargs": nargs,

        "Metadata":
        """To reload the objects, create a python script under the result saving directory, and use the following code:\nimport pickle\nobjects_pkl_save_path = "your/path/to/the/file.pkl"\nobject_pkl_dict = pickle.load(open(objects_pkl_save_path, 'rb'))"""
    }
    with open(objects_pkl_save_path, 'wb') as f:
        pickle.dump(objects_pkl_dict, f) # Save the result dictionary
        print(f"Saved objects pkl file to {objects_pkl_save_path}")
    return