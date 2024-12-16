"""
一些残存的工具函数，暂时没用
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import time
import tqdm






def visualize_var(Y_pred,Y_true,var_idx,var_names,var_units,
                    data_name="0",
                    var_mean=None,
                    var_std_dev=None,
                    plot_residual=False,
                    rescale=False
                    ):
    '''
    Y_pred: numpy array of shape (n_samples, n_vars)
    Y_true: numpy array of shape (n_samples, n_vars)
    var_idx: index of the variable to be visualized
    
    plot_residual: whether to plot the residual or the actual values
    rescale: whether to rescale the values to their original scale or not
    '''
    assert type(Y_pred)==np.ndarray and Y_pred.ndim==2, "Y_pred should be a 2D numpy array"
    assert type(Y_true)==np.ndarray and Y_true.ndim==2, "Y_true should be a 2D numpy array"
    assert Y_pred.shape==Y_true.shape, "Y_pred and Y_true should have the same shape"
    assert len(var_names)==len(var_units), "var_names and var_units should have the same length"
    assert var_idx in range(len(var_names)), "var_idx should be within the range of var_names"
    assert plot_residual in [True, False], "plot_residual should be either True or False"
    assert rescale in [True, False], "rescale should be either True or False"

    try:
        from matplotlib.font_manager import FontProperties
        font1 = FontProperties(fname=r"C:\\Windows\\Fonts\\STFANGSO.ttf", size=14)
        font2 = FontProperties(fname=r"C:\\Windows\\Fonts\\STFANGSO.ttf", size=12)
        font3 = FontProperties(fname=r"C:\\Windows\\Fonts\\STFANGSO.ttf", size=10)
        font4 = FontProperties(fname=r"C:\\Windows\\Fonts\\STFANGSO.ttf", size=7)
    except:
        raise Exception('为了中文的正常显示，请确保您的系统为Windows系统，并安装了STFANGSO.ttf字体。\n通常该字体的路径为"C:\\Windows\\Fonts\\STFANGSO.ttf"')

    y_true=Y_true[:,var_idx] # shape: (n_samples,)
    y_pred=Y_pred[:,var_idx] # shape: (n_samples,)
    y_mean=var_mean[var_idx] # float
    y_std_dev=var_std_dev[var_idx] # float
    plt.figure(figsize=(12,5))
    
    if plot_residual==False:
        if rescale == False:
            plt.plot(y_true,c='b',label='True')
            plt.plot(y_pred,c='r',label='Predicted')
            plt.ylabel("Normalized Value")
        elif rescale == True:
            plt.plot(y_true*y_std_dev+y_mean,c='b',label='True')
            plt.plot(y_pred*y_std_dev+y_mean,c='r',label='Predicted')
            plt.ylabel(var_units[var_idx])
    elif plot_residual==True:
        plt.axhline(y=0)
        if rescale == False:
            plt.plot(y_true-y_pred,c='b',label='Residual')
            plt.ylabel("Normalized Value")
        elif rescale == True:
            plt.plot((y_true-y_pred)*y_std_dev,c='b',label='Residual')
            plt.ylabel(var_units[var_idx])

    title_str="Prediction of {} on {}".format(var_names[var_idx],data_name)
    if rescale == False:
        title_str+="\nRMSE: {:.6f}".format(np.sqrt(((y_true-y_pred)**2).mean()))
        title_str+="\nMAE: {:.6f}".format(np.abs(y_true-y_pred).mean())
    elif rescale == True:
        title_str+="\nRMSE: {:.6f}".format(np.sqrt((((y_true-y_pred)*y_std_dev)**2).mean()))
        title_str+="\nMAE: {:.6f}".format(np.abs((y_true-y_pred)*y_std_dev).mean())
    plt.title(title_str,fontproperties=font2)
    
    plt.xlabel('Time')
    plt.legend()
    plt.show()










