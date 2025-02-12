import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

class ModelTest:
    def __init__(self, model, dataset,
                    device='cpu'
                    ):
        self.model=model # A neural network model
        self.dataset=dataset
        self.device=device
        self.Y_true=None
        self.Y_pred=None # To be set by `get_pred_true_pairs()`


    def predict(self, X, with_Tensor=False):
        r'''
        Predict on a given series data and return the predicted values in numpy array format.
        
        :param model: the model to be used
        :param X: numpy array of shape (n_subseries, input_len, n_input_vars)
        '''
        assert type(X)==np.ndarray and X.ndim==3, "X should be a 3D numpy array"
        assert X.shape[1]==self.dataset.input_len, "Each subseries in X should have the same length as input_len in self.dataset"
        assert X.shape[2]==self.dataset.n_input_vars, "Each subseries in X should have the same number of variables (channels) as in self.dataset"

        n_output_vars=self.dataset.n_output_vars

        if with_Tensor:
            X=torch.Tensor(X).to(self.device)
            Y_pred_Tensor=self.model(X).reshape(-1,n_output_vars) # concatenate the predictions by time order
            Y_pred=Y_pred_Tensor.detach().cpu().numpy()
            
        elif not with_Tensor:
            Y_pred=self.model(X).reshape(-1,n_output_vars) # concatenate the predictions by time order

        else:
            raise ValueError("`with_Tensor` should be either True or False")
        
        self.Y_pred=Y_pred
        return Y_pred


    def get_pred_true_pairs(self, X, Y, with_Tensor=False):
        '''
        Predict on a given series data and return the predicted and true values.
        Usually used when self.model is a torch.nn.Module.

        :param X: numpy array of shape (n_subseries, input_len, n_input_vars)
        :param Y: numpy array of shape (n_subseries, output_len, n_output_vars)

        :return: Y_pred: numpy array of shape (n_timesteps, n_output_vars)
        :return: Y_true: numpy array of shape (n_timesteps, n_output_vars)
        '''
        assert type(X)==np.ndarray and X.ndim==3, "X should be a 3D numpy array"
        assert type(Y)==np.ndarray and Y.ndim==3, "Y should be a 3D numpy array"
        assert X.shape[0]==Y.shape[0], "X and Y should have the same number of samples"

        n_output_vars=self.dataset.n_output_vars
        Y_pred=self.predict(X, with_Tensor=with_Tensor) # shape: (n_timesteps, n_output_vars)
        Y_true=Y.reshape(-1,n_output_vars) # concatenate the true values by time order
        self.Y_pred, self.Y_true=Y_pred, Y_true
        return Y_pred, Y_true # shape: (n_timesteps, n_output_vars)


    def plot_all_predictions(self, Y_pred, Y_true,
                                plot_residual=False,
                                rescale=False,
                                suptitle_text="All Predictions",
                                n_columns=3, # The number of columns in the subplot grid
                                figsize=(12,4),
                                save_path=None,
                                ):
        r'''
        
        :param Y_pred: numpy array of shape (n_timesteps, n_output_vars)
        :param Y_true: numpy array of shape (n_timesteps, n_output_vars)
        '''
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
        plt.figure(figsize=figsize)
        for var_idx , (var_name, var_unit, y_mean, y_std_dev) \
            in enumerate(zip(
                    self.dataset.output_var_names,
                    self.dataset.output_var_units,
                    self.dataset.output_var_mean,
                    self.dataset.output_var_std_dev
                    )):
            plt.subplot(n_output_vars//n_columns+1, n_columns, var_idx+1)
            y_true=Y_true[:,var_idx] # shape: (n_samples,)
            y_pred=Y_pred[:,var_idx] # shape: (n_samples,)
            if plot_residual==False:
                #if var_name == "烟气含氧量（CEMS）":
                #    plt.ylim((0,10))
                #    print(f"The lower bound of {var_name} is reset")
                #if var_name == "NOX浓度":
                #    plt.ylim((0,25))
                #    print(f"The upper bound of {var_name} is reset")
                #if var_name == "一氧化碳":
                #    plt.ylim((0,2))
                #    print(f"The upper bound of {var_name} is reset")
                #if var_name == "烟气湿度（CEMS）":
                #    plt.ylim((0,10))
                #    print(f"The upper bound of {var_name} is reset") 
                #if var_name == "烟气压力（CEMS）":
                #    plt.ylim((-40,0))
                #    print(f"The upper bound of {var_name} is reset")
                #if var_name == "烟气压力（CEMS）":
                #    pass
                #    #plt.ylim(())
                #    #print(f"The upper bound of {var_name} is set to 0")
                #if var_name == "炉膛出口烟气压力":
                #    plt.ylim((0,3))
                #    print(f"The upper bound of {var_name} is reset")

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
        plt.tight_layout() # 调整子图间距，防止重叠
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight') # 保存图片
            print(f"Figure of model prediction saved to {save_path}")
        plt.show()


    def get_prediction_info(self,X_test_grouped,Y_test_grouped,
                            with_Tensor=False
                            ):
        r'''
        :param X_test_grouped: list of numpy arrays of shape (n_subseries, input_len, n_input_vars)
        :param Y_test_grouped: list of numpy arrays of shape (n_subseries, output_len, n_output_vars)
        :return: pandas DataFrame of shape (n_output_vars, 5)
        '''
        assert type(X_test_grouped)==list and type(Y_test_grouped)==list, "X_test_grouped and Y_test_grouped should be lists"
        assert len(X_test_grouped)==len(Y_test_grouped), "X_test_grouped and Y_test_grouped should have the same length"
        
        var_prediction_info={}
        n_groups=len(X_test_grouped)
        SSE=0 # Sum of Squared Errors
        SAE=0 # Sum of Absolute Errors
        total_timesteps=0
        for var_idx , (var_name, var_unit, y_mean, y_std_dev) \
                in enumerate(zip(
                    self.dataset.output_var_names,
                    self.dataset.output_var_units,
                    self.dataset.output_var_mean,
                    self.dataset.output_var_std_dev
                    )):
            for data_idx in range(n_groups):
                Y_pred, Y_true=self.get_pred_true_pairs(X_test_grouped[data_idx],Y_test_grouped[data_idx],with_Tensor=with_Tensor)
                y_true=Y_true[:,var_idx] # shape: (n_timesteps,)
                y_pred=Y_pred[:,var_idx] # shape: (n_timesteps,)
                SSE+=((y_true-y_pred)**2).sum()
                SAE+=np.abs(y_true-y_pred).sum()
                total_timesteps+=y_true.shape[0]
            RMSE=np.sqrt(SSE/total_timesteps) # standard RMSE
            MAE=SAE/total_timesteps # standard MAE
            RMSE_rescaled=RMSE*y_std_dev
            MAE_rescaled=MAE*y_std_dev
            var_prediction_info[var_name]=[var_unit, RMSE_rescaled, RMSE, MAE_rescaled, MAE]

        return pd.DataFrame(var_prediction_info,index=['unit', 'RMSE', 'RMSE_standard','MAE', 'MAE_standard',]).T