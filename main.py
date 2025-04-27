import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import torch
import os
import sys
import time
import json
import random
import argparse
import pickle
import traceback
from functools import partial

matplotlib.use('Agg') # Set the backend to disable figure display window
warnings.filterwarnings("ignore", category=UserWarning) # To filter the warning of disabling plt.show()

try:
    from matplotlib.font_manager import FontProperties
    font1 = FontProperties(fname=r"C:\\Windows\\Fonts\\STFANGSO.ttf", size=14)
    font2 = FontProperties(fname=r"C:\\Windows\\Fonts\\STFANGSO.ttf", size=12)
    font3 = FontProperties(fname=r"C:\\Windows\\Fonts\\STFANGSO.ttf", size=10)
    font4 = FontProperties(fname=r"C:\\Windows\\Fonts\\STFANGSO.ttf", size=7)
except:
    raise Exception('为了中文的正常显示，请确保您的系统为Windows系统，并安装了STFANGSO.ttf字体。\n通常该字体的路径为"C:\\Windows\\Fonts\\STFANGSO.ttf"')



# Default parameter dictionary
param_dict = {
    "model_name": "CNNLSTM",
    "data_path": r"E:\PythonProjects\gas-data-prediction\data\太原-wavelet-gaussian.xlsx",
    "seed": 12345678,
    
    # Dataset Settings
    "input_var_names": [
        "Environment Temperature",
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
        #"Flame Temperature",
        #"Flame Speed",
    ], # The names of input variables.
    "output_var_names": [
        #"O2",
        #"CO2",
        "NOx",
        #"Smoke Temperature",
    ], # The names of output variables.
    "input_len": 30, # The temporal length of input sequence.
    "output_len": 1, # The temporal length of output sequence.
    "overlap": 1, # The overlap between input and output sequences.
    "pool_size": 5, # TODO
    "transform": True, # Whether to transform the input and output variables.
    "transform_slope": 3.0,
    "transform_l_threshold": 25.0,
    "transform_u_threshold": 35.0,
    "transform_var_name": "NOx",

    # Saving Settings
    "save_result": True, # Whether to save the result files.
    "result_dir": "./results0427", # The directory to save all results. The results of each single run will be saved in a sub-directory under this directory.
    "experiment_dir_suffix": "", # The suffix of the experiment directory name.
    "comment": "default", # Supplementary information of the experiment.
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
    "device": "cuda" if torch.cuda.is_available() else "cpu", # device to train the neural network model. Options: ["cuda", "cpu"].
    "optimizer_name": "Adam", # The optimizer used for training. Options: ["Adam", "SGD", "Adagrad", "Adadelta", "RMSprop"].
    "learning_rate": 0.001, # learning rate
    "batch_size": 64, # batch size
    "num_epochs": 100, # maximum number of epochs (may stop earlier due to early stopping settings)
}



def main(args):

    try:
        
        # Set another argument namespace to store intermediate variables (variables that are not directly input, but can be calculated from the input)
        nargs = argparse.Namespace() # New arguments; To be updated later

        # Set up the experiment directory
        if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir)
            print("Result directory created at {}.".format(args.result_dir))

        time_string=str(int(time.time())) # Use current timestamp to identify the experiment
        setattr(nargs, 'time_string', time_string)

        experiment_dir=os.path.join(args.result_dir, 'experiment_{}_{}_{}_{}_{}_{}'.format(
                time_string, args.model_name, args.input_len, args.output_len, args.overlap, args.pool_size))
        if args.experiment_dir_suffix:
            experiment_dir += '_' + args.experiment_dir_suffix
        setattr(nargs, 'experiment_dir', experiment_dir)

        if args.save_result and not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
            print("Experiment directory created at {}.".format(experiment_dir))

        # Set the logger
        from debug.logger import get_logger, close_logger
        log_file_path = os.path.join(experiment_dir, 'log.log')
        logger = get_logger(log_file_path=log_file_path)

        # Save the experiment configuration
        from utils.format import NumpyEncoder, namespace2dict
        config_dict = namespace2dict(args)
        config_save_path = os.path.join(experiment_dir, 'config.json')
        json.dump(config_dict, open(config_save_path, 'w'), indent=4, cls=NumpyEncoder)
        logger.info("Experiment config saved to '{}'.".format(config_save_path))

    except Exception as e:
        print("Error: {}".format(e))
        return

    try: # Now the logger has been successfully set up, and errors can be logged in the log file.

        from utils import set_seed
        set_seed(args.seed)

        # Determine model type
        from utils import get_model_type
        model_type = get_model_type(args.model_name)
        setattr(nargs, 'model_type', model_type)

        print("----------------Model Setting Information----------------")
        logger.info("Model name: {}".format(args.model_name))
        logger.info("Input variables: {}".format(args.input_var_names))
        logger.info("Output variables: {}".format(args.output_var_names))
        logger.info("Input length: {}".format(args.input_len))
        logger.info("Output length: {}".format(args.output_len))
        logger.info("Input channels: {}".format(len(args.input_var_names)))
        logger.info("Output channels: {}".format(len(args.output_var_names)))
        logger.info("Overlap: {}".format(args.overlap))
        logger.info("Result directory: {}".format(args.result_dir))
        logger.info("Seed: {}".format(args.seed))
        print("---------------------------------------------------------")

        
        # ----------------------------------------------------------------------------------------------
        # Load the Preprocessed Dataset
        from components import get_dataset
        (X_train, Y_train), \
        (X_test, Y_test), \
        (X_train_grouped, Y_train_grouped), \
        (X_test_grouped, Y_test_grouped), \
        (input_var_mean, output_var_mean), \
        (input_var_std_dev, output_var_std_dev), \
        (input_var_indices, output_var_indices), \
        (input_var_units, output_var_units) = get_dataset(args, logger)

        setattr(nargs, 'input_var_units', input_var_units)
        setattr(nargs, 'input_var_mean', input_var_mean)
        setattr(nargs, 'input_var_std_dev', input_var_std_dev)
        setattr(nargs, 'output_var_units', output_var_units)
        setattr(nargs, 'output_var_mean', output_var_mean)
        setattr(nargs, 'output_var_std_dev', output_var_std_dev)


        # ----------------------------------------------------------------------------------------------
        # Get the trained model
        from components import model_building_and_training
        model = model_building_and_training(args, nargs, X_train, Y_train, logger)


        # ----------------------------------------------------------------------------------------------
        # Get prediction information
        from utils.model_test import ModelTest
        from utils.transform import transform, inverse_transform
        
        Exp=ModelTest(model=model, device=args.device) # Experiment object initialization
        with_Tensor=True if nargs.model_type=='NN' else False
        var_prediction_info=[] # A list where each element is a dictionary containing information about a single output variable's prediction performance.
        SSE=0 # Sum of Squared Errors
        SAE=0 # Sum of Absolute Errors
        
        for var_idx , (var_name, var_unit, y_mean, y_std_dev) \
            in enumerate(zip(
                args.output_var_names,
                nargs.output_var_units,
                nargs.output_var_mean,
                nargs.output_var_std_dev
                )): # Get prediction info for each output variable
            inverse_transform_func=lambda x: x
            if args.transform and var_name == args.transform_var_name:
                var_idx = args.output_var_names.index(args.transform_var_name)
                inverse_transform_func=partial(inverse_transform, slope=args.transform_slope,
                                                l_threshold=args.transform_l_threshold, u_threshold=args.transform_u_threshold, var_idx=var_idx)

            total_timesteps=0
            for data_idx in range(len(X_test_grouped)):
                Y_pred, Y_true=Exp.get_pred_true_series_pairs(X_test_grouped[data_idx],Y_test_grouped[data_idx],with_Tensor=with_Tensor)
                y_true=Y_true[:,var_idx] # Shape: (n_timesteps,)
                y_pred=Y_pred[:,var_idx] # Shape: (n_timesteps,)
                y_true_rescaled=inverse_transform_func(y_true*y_std_dev+y_mean) # Shape: (n_timesteps,)
                y_pred_rescaled=inverse_transform_func(y_pred*y_std_dev+y_mean) # Shape: (n_timesteps,)
                SSE+=((y_true-y_pred)**2).sum()
                SAE+=np.abs(y_true-y_pred).sum()
                SSE_rescaled=((y_true_rescaled-y_pred_rescaled)**2).sum()
                SAE_rescaled=np.abs(y_true_rescaled-y_pred_rescaled).sum()
                total_timesteps+=y_true.shape[0]
            RMSE=np.sqrt(SSE/total_timesteps) # standard RMSE
            MAE=SAE/total_timesteps # standard MAE
            RMSE_rescaled=np.sqrt(SSE_rescaled/total_timesteps) # rescaled RMSE
            MAE_rescaled=SAE_rescaled/total_timesteps # rescaled MAE
            var_prediction_info.append({
                'var_name': var_name,
                'var_unit': var_unit,
                'RMSE': RMSE,
                'RMSE_rescaled': RMSE_rescaled,
                'MAE': MAE,
                'MAE_rescaled': MAE_rescaled,
            })
        

        # ----------------------------------------------------------------------------------------------
        # Plot the predictions and residuals
        if args.save_plots is True:

            Y_pred, Y_true=Exp.get_pred_true_series_pairs(X_test_grouped[0],Y_test_grouped[0],with_Tensor=with_Tensor) # Plot on the first time series in test data
            n_columns = 2
            for plot_name, (plot_residual, rescale) in zip(["pred_std","pred","res_std","res"], [(0,0), (0,1), (1,0), (1,1)]):
                # Save four plots:
                # - pred_std: standardized prediction
                # - pred: rescaled prediction
                # - res_std: standardized residual
                # - res: rescaled residual
                if inverse_transform_func is None:
                    inverse_transform_func=lambda x: x # identity map

                plt.figure(figsize=(12,8))
                for var_idx , (var_name, var_unit, y_mean, y_std_dev) \
                    in enumerate(zip(
                            args.output_var_names,
                            nargs.output_var_units,
                            nargs.output_var_mean,
                            nargs.output_var_std_dev
                            )):
                    inverse_transform_func=lambda x: x
                    if args.transform and var_name == args.transform_var_name:
                        var_idx = args.output_var_names.index(args.transform_var_name)
                        inverse_transform_func=partial(inverse_transform, slope=args.transform_slope,
                                                        l_threshold=args.transform_l_threshold, u_threshold=args.transform_u_threshold, var_idx=var_idx)
                    y_true=Y_true[:,var_idx] # Shape: (n_samples,)
                    y_pred=Y_pred[:,var_idx] # Shape: (n_samples,)
                    y_true_rescaled=inverse_transform_func(y_true*y_std_dev+y_mean) # Shape: (n_samples,)
                    y_pred_rescaled=inverse_transform_func(y_pred*y_std_dev+y_mean) # Shape: (n_samples,)

                    plt.subplot(len(args.output_var_names)//n_columns+1, n_columns, var_idx+1)
                    if plot_residual==False:
                        if rescale == False:
                            plt.plot(y_true,c='b',label='True')
                            plt.plot(y_pred,c='r',label='Predicted')
                            plt.ylabel("Normalized Value")
                        elif rescale == True:
                            plt.plot(y_true_rescaled,c='b',label='True')
                            plt.plot(y_pred_rescaled,c='r',label='Predicted')
                            plt.ylabel(var_unit)
                    elif plot_residual==True:
                        plt.axhline(y=0)
                        if rescale == False:
                            plt.plot(y_true-y_pred,c='b',label='Residual')
                            plt.ylabel("Normalized Value")
                        elif rescale == True:
                            plt.plot(y_true_rescaled-y_pred_rescaled,c='b',label='Residual')
                            plt.ylabel(var_unit)

                    title_str=f"{var_name}"
                    if rescale == False:
                        title_str+="\nRMSE: {:.6f}".format(np.sqrt(((y_true-y_pred)**2).mean()))
                        title_str+="\nMAE: {:.6f}".format(np.abs(y_true-y_pred).mean())
                    elif rescale == True:
                        title_str+="\nRMSE: {:.6f}".format(np.sqrt(((y_true_rescaled-y_pred_rescaled)**2).mean()))
                        title_str+="\nMAE: {:.6f}".format(np.abs(y_true_rescaled-y_pred_rescaled).mean())
                    plt.title(title_str,fontproperties=font2)
                    plt.xlabel('Time')
                    plt.legend()

                plt.suptitle("All predictions",fontproperties=font1)
                plt.tight_layout() # Adjust subplot spacing to avoid overlap
                fig_save_path = os.path.join(nargs.experiment_dir, '{}_{}.png'.format(args.model_name, plot_name))
                plt.savefig(fig_save_path, dpi=200, bbox_inches='tight')
                print(f"Figure of model prediction saved to {fig_save_path}")
                plt.close()


        # ----------------------------------------------------------------------------------------------
        # Save Results
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
                "pool_size": args.pool_size,
                "input_var_names": args.input_var_names,
                "output_var_names": args.output_var_names,
            },
            "prediction_info": var_prediction_info,
            "hyperparameters": hyperparameter_dict,
            "seed": args.seed,
            "comment": args.comment,
        }

        result_save_path = os.path.join(nargs.experiment_dir, 'result.json')
        with open(result_save_path, 'w') as f:
            json.dump(result_dict, f, indent=4, cls=NumpyEncoder)
        logger.info("Result saved to '{}'.".format(result_save_path))

        # ----------------------------------------------------------------------------------------------
        # Save objects
        if args.save_objects is True:
            objects_pkl_save_path = os.path.join(nargs.experiment_dir, 'objects.pkl')
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



        close_logger(logger)
        return var_prediction_info

    except Exception as e:
        logger.error("Error: {}".format(e))
        logger.error(traceback.format_exc())
        close_logger(logger)








if __name__ == '__main__':
    t0=time.time() # Start the timer
    args = argparse.Namespace(**param_dict) # Convert the dictionary to a namespace object for easier access to the parameters.
    main(args)

