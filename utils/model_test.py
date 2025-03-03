import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch


class ModelTest:
    r"""
    A class for testing a prediction model.

    The model (`self.model`) can be any model that takes input X and gives output Y, where:

    - The input X is a 3-d numpy array of shape (N, input_len, n_input_vars),
    - The output Y is a 3-d numpy array of shape (N, output_len, n_output_vars).

    X or Y represent a segment of input or output series data.
    
    - The input series data has `input_len` time steps and `n_input_vars` variables,
    - The output series data has `output_len` time steps and `n_output_vars` variables,
    - `N` is the number of samples in the batch.
    """
    def __init__(self, model, device='cpu',
                    ):
        self.model=model # A prediction model that takes input X and gives output Y
        self.device=device

    def predict(self, X, with_Tensor=False):
        r"""
        Predict on a given series data and return the predicted values in numpy array format.
        
        :param X: numpy array of shape (n_subseries, input_len, n_input_vars)
        """
        assert type(X)==np.ndarray and X.ndim==3, "X should be a 3D numpy array"

        if with_Tensor is True: # If the model requires tensor input
            X_Tensor=torch.Tensor(X).to(self.device)
            Y_pred_Tensor=self.model(X_Tensor)
            _, _, n_output_vars=Y_pred_Tensor.shape
            Y_pred=Y_pred_Tensor.detach().cpu().numpy().reshape(-1,n_output_vars) # concatenate the predictions by time order
            
        elif with_Tensor is False: # If the model requires numpy input
            Y_pred=self.model(X)
            _, _, n_output_vars=Y_pred.shape
            Y_pred=Y_pred.reshape(-1,n_output_vars) # concatenate the predictions by time order

        else:
            raise ValueError("`with_Tensor` should be either True or False")

        return Y_pred


    def get_pred_true_series_pairs(self, X, Y, with_Tensor=False):
        r"""
        Predict on a given series data and return the predicted and true values.
        Usually used when all data in X, Y come from the same series.

        :param X: numpy array of shape (n_subseries, input_len, n_input_vars)
        :param Y: numpy array of shape (n_subseries, output_len, n_output_vars)

        :return Y_pred, Y_true:

        - Y_pred: numpy array of shape (n_timesteps, n_output_vars)
        - Y_true: numpy array of shape (n_timesteps, n_output_vars)
        """
        assert type(X)==np.ndarray and X.ndim==3, "X should be a 3D numpy array"
        assert type(Y)==np.ndarray and Y.ndim==3, "Y should be a 3D numpy array"
        assert X.shape[0]==Y.shape[0], "X and Y should have the same number of samples"

        n_output_vars=Y.shape[2]
        Y_pred=self.predict(X, with_Tensor=with_Tensor) # shape: (n_timesteps, n_output_vars)
        Y_true=Y.reshape(-1,n_output_vars) # concatenate the true values by time order
        return Y_pred, Y_true # shape: (n_timesteps, n_output_vars)


    def plot_all_predictions(self, Y_pred, Y_true, output_var_names, output_var_units, output_var_mean, output_var_std_dev,
                                plot_residual=False,
                                rescale=False,
                                suptitle_text="All Predictions",
                                n_columns=2, # The number of columns in the subplot grid
                                figsize=(12,4),
                                save_path=None,
                                ):
        r"""
        Visualize the predicted and true values for all output variables.

        :param Y_pred: numpy array of shape `(n_timesteps, n_output_vars)`.
        :param Y_true: numpy array of shape `(n_timesteps, n_output_vars)`.
        :param output_var_names: list of strings, the names of the output variables.
        :param output_var_units: list of strings, the units of the output variables.
        :param output_var_mean: list or other iterable of floats, the means of the output variables.
        :param output_var_std_dev: list or other iterable of floats, the standard deviations of the output variables.

        :param plot_residual: bool, whether to plot the residual (`True`) or the absolute error (`False`).
        :param rescale: bool, whether to rescale the values to the original scale.
        :param suptitle_text: str, the text for the suptitle of the plot.
        :param n_columns: int, the number of columns in the subplot grid.
        :param figsize: tuple, the size of the figure.
        :param save_path: str, the path to save the figure. If `None`, the figure will not be saved.
        """
        try:
            from matplotlib.font_manager import FontProperties
            font1 = FontProperties(fname=r"C:\\Windows\\Fonts\\STFANGSO.ttf", size=14)
            font2 = FontProperties(fname=r"C:\\Windows\\Fonts\\STFANGSO.ttf", size=12)
            font3 = FontProperties(fname=r"C:\\Windows\\Fonts\\STFANGSO.ttf", size=10)
            font4 = FontProperties(fname=r"C:\\Windows\\Fonts\\STFANGSO.ttf", size=7)
        except:
            raise Exception('为了中文的正常显示，请确保您的系统为Windows系统，并安装了STFANGSO.ttf字体。\n通常该字体的路径为"C:\\Windows\\Fonts\\STFANGSO.ttf"')

        assert type(n_columns)==int and n_columns>0, "n_columns should be a positive integer"
        n_output_vars=Y_pred.shape[1]
        assert len(output_var_names)==len(output_var_units)==len(output_var_mean)==len(output_var_std_dev)==n_output_vars, \
            "The length of `output_var_names`, `output_var_units`, `output_var_mean`, and `output_var_std_dev` should all equal to the number of output variables in the dataset."
        plt.figure(figsize=figsize)
        for var_idx , (var_name, var_unit, y_mean, y_std_dev) \
            in enumerate(zip(
                    output_var_names,
                    output_var_units,
                    output_var_mean,
                    output_var_std_dev
                    )):
            plt.subplot(n_output_vars//n_columns+1, n_columns, var_idx+1)
            y_true=Y_true[:,var_idx] # shape: (n_samples,)
            y_pred=Y_pred[:,var_idx] # shape: (n_samples,)
            if plot_residual==False:
                if rescale == False:
                    plt.plot(y_true,c='b',label='True')
                    plt.plot(y_pred,c='r',label='Predicted')
                    plt.ylabel("Normalized Value")
                elif rescale == True:
                    plt.plot(y_true*y_std_dev+y_mean,c='b',label='True')
                    plt.plot(y_pred*y_std_dev+y_mean,c='r',label='Predicted')
                    plt.ylabel(var_unit)
            elif plot_residual==True:
                plt.axhline(y=0)
                if rescale == False:
                    plt.plot(y_true-y_pred,c='b',label='Residual')
                    plt.ylabel("Normalized Value")
                elif rescale == True:
                    plt.plot((y_true-y_pred)*y_std_dev,c='b',label='Residual')
                    plt.ylabel(var_unit)

            title_str=f"{var_name}"
            if rescale == False:
                title_str+="\nRMSE: {:.6f}".format(np.sqrt(((y_true-y_pred)**2).mean()))
                title_str+="\nMAE: {:.6f}".format(np.abs(y_true-y_pred).mean())
            elif rescale == True:
                title_str+="\nRMSE: {:.6f}".format(np.sqrt((((y_true-y_pred)*y_std_dev)**2).mean()))
                title_str+="\nMAE: {:.6f}".format(np.abs((y_true-y_pred)*y_std_dev).mean())
            plt.title(title_str,fontproperties=font2)

            plt.xlabel('Time')
            plt.legend()

        plt.suptitle(suptitle_text,fontproperties=font1)
        plt.tight_layout() # Adjust subplot spacing to avoid overlap
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight') # Save
            print(f"Figure of model prediction saved to {save_path}")
        plt.show()


    def get_prediction_info(self, X_test_grouped, Y_test_grouped, output_var_names, output_var_units, output_var_mean, output_var_std_dev,
                            with_Tensor=False
                            ):
        r"""
        :param X_test_grouped: list of numpy arrays of shape (n_subseries, input_len, n_input_vars).
        :param Y_test_grouped: list of numpy arrays of shape (n_subseries, output_len, n_output_vars)

            -  Each element in the list corresponds to a group of subseries that come from a same series.

        :param output_var_names: list of strings, the names of the output variables.
        :param output_var_units: list of strings, the units of the output variables.
        :param output_var_mean: list or other iterable of floats, the means of the output variables.
        :param output_var_std_dev: list or other iterable of floats, the standard deviations of the output variables.
        :param with_Tensor: bool, whether to activate tensor operation for the model. Set to `True` if the model is a PyTorch model.
        :return var_prediction_info: list of dictionaries containing the prediction information for each output variable.
        """
        assert type(X_test_grouped)==list and type(Y_test_grouped)==list, "`X_test_grouped` and `Y_test_grouped` should be lists"
        assert len(X_test_grouped)==len(Y_test_grouped), "`X_test_grouped` and `Y_test_grouped` should have the same length"
        assert len(output_var_names)==len(output_var_units)==len(output_var_mean)==len(output_var_std_dev), \
            "The length of `output_var_names`, `output_var_units`, `output_var_mean`, and `output_var_std_dev` should be the same"
        assert len(output_var_names)==Y_test_grouped[0].shape[2], "The number of output variables in `output_var_names` should match the number of output variables in the given dataset"

        var_prediction_info=[]
        n_groups=len(X_test_grouped)
        SSE=0 # Sum of Squared Errors
        SAE=0 # Sum of Absolute Errors
        total_timesteps=0
        for var_idx , (var_name, var_unit, y_mean, y_std_dev) \
                in enumerate(zip(
                    output_var_names,
                    output_var_units,
                    output_var_mean,
                    output_var_std_dev
                    )):
            for data_idx in range(n_groups):
                Y_pred, Y_true=self.get_pred_true_series_pairs(X_test_grouped[data_idx],Y_test_grouped[data_idx],with_Tensor=with_Tensor)
                y_true=Y_true[:,var_idx] # shape: (n_timesteps,)
                y_pred=Y_pred[:,var_idx] # shape: (n_timesteps,)
                SSE+=((y_true-y_pred)**2).sum()
                SAE+=np.abs(y_true-y_pred).sum()
                total_timesteps+=y_true.shape[0]
            RMSE=np.sqrt(SSE/total_timesteps) # standard RMSE
            MAE=SAE/total_timesteps # standard MAE
            RMSE_rescaled=RMSE*y_std_dev
            MAE_rescaled=MAE*y_std_dev
            var_prediction_info.append({
                'var_name': var_name,
                'var_unit': var_unit,
                'RMSE': RMSE,
                'RMSE_rescaled': RMSE_rescaled,
                'MAE': MAE,
                'MAE_rescaled': MAE_rescaled,
            })

        return var_prediction_info