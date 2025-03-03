"""
Utility functions for Exploratory Data Analysis (EDA)
未完善，暂时没用，不要看
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_file(file_path,
            var_names=['% O2', 'ppm CO', '% CO2', 'ppm NO', 'ppm NO2', '°C 烟温', 'ppm NOx', 'ppm SO2', '°C 环温', 'l/min 泵流量'],
            figsize=(16, 12)
            ):
    """
    Plot all timeseries data of a file.
    """
    try:
        from matplotlib.font_manager import FontProperties
        font1 = FontProperties(fname=r"C:\\Windows\\Fonts\\STFANGSO.ttf", size=14)
        font2 = FontProperties(fname=r"C:\\Windows\\Fonts\\STFANGSO.ttf", size=12)
        font3 = FontProperties(fname=r"C:\\Windows\\Fonts\\STFANGSO.ttf", size=10)
        font4 = FontProperties(fname=r"C:\\Windows\\Fonts\\STFANGSO.ttf", size=7)
    except:
        raise Exception("请确保您的系统为Windows系统，并安装了STFANGSO.ttf字体")

    df=pd.read_excel(file_path)[var_names].apply( # 读取指定变量（var_names）的数据
            lambda x:pd.to_numeric(x, errors='coerce') # 对于非数字值，用nan填充
            ).fillna(0) # 对于nan值，用0填充
    plt.figure(figsize=figsize)
    plt.suptitle('Variable Time Series of {}'.format(
                    os.path.basename(file_path))
                 , fontproperties=font2)

    nrows=len(var_names)//3+1
    ncolumns=3
    length=df.shape[0]
    for i, var_name in enumerate(var_names):
        plt.subplot(nrows, ncolumns, i + 1)
        plt.plot(np.arange(length), df[var_name])
        plt.title(var_name, fontproperties=font2)
    plt.tight_layout()
    plt.show()



def plot_var(file_path, var_name,
            figsize=(8, 6)
            ):
    """
    Plot the timeseries data of a single variable within a file.
    """
    try:
        from matplotlib.font_manager import FontProperties
        font1 = FontProperties(fname=r"C:\\Windows\\Fonts\\STFANGSO.ttf", size=14)
        font2 = FontProperties(fname=r"C:\\Windows\\Fonts\\STFANGSO.ttf", size=12)
        font3 = FontProperties(fname=r"C:\\Windows\\Fonts\\STFANGSO.ttf", size=10)
        font4 = FontProperties(fname=r"C:\\Windows\\Fonts\\STFANGSO.ttf", size=7)
    except:
        raise Exception("请确保您的系统为Windows系统，并安装了STFANGSO.ttf字体")

    df=pd.read_excel(file_path)[[var_name]].apply( # 读取指定变量（var_names）的数据
            lambda x:pd.to_numeric(x, errors='coerce') # 对于非数字值，用nan填充
            ).fillna(0) # 对于nan值，用0填充
    plt.figure(figsize=figsize)
    plt.suptitle('Variable Time Series of {}'.format(
                    os.path.basename(file_path))
                 , fontproperties=font2)
    plt.plot(np.arange(df.shape[0]), df[var_name])
    plt.title(var_name, fontproperties=font2)
    plt.tight_layout()
    plt.show()



def hist_file(file_path,
            var_names=['% O2', 'ppm CO', '% CO2', 'ppm NO', 'ppm NO2', '°C 烟温', 'ppm NOx', 'ppm SO2', '°C 环温', 'l/min 泵流量'],
            bins=100,
            figsize=(16, 12)
            ):
    """
    Histogram of all timeseries data within a file.
    """
    try:
        from matplotlib.font_manager import FontProperties
        font1 = FontProperties(fname=r"C:\\Windows\\Fonts\\STFANGSO.ttf", size=14)
        font2 = FontProperties(fname=r"C:\\Windows\\Fonts\\STFANGSO.ttf", size=12)
        font3 = FontProperties(fname=r"C:\\Windows\\Fonts\\STFANGSO.ttf", size=10)
        font4 = FontProperties(fname=r"C:\\Windows\\Fonts\\STFANGSO.ttf", size=7)
    except:
        raise Exception("请确保您的系统为Windows系统，并安装了STFANGSO.ttf字体")

    df=pd.read_excel(file_path)[var_names].apply( # 读取指定变量（var_names）的数据
            lambda x:pd.to_numeric(x, errors='coerce') # 对于非数字值，用nan填充
            ).fillna(0) # 对于nan值，用0填充
    plt.figure(figsize=figsize)
    plt.suptitle('Histogram of Variables of {}'.format(
                    os.path.basename(file_path))
                 , fontproperties=font2)

    nrows=len(var_names)//3+1
    ncolumns=3
    length=df.shape[0]
    for i, var_name in enumerate(var_names):
        plt.subplot(nrows, ncolumns, i + 1)
        plt.hist(df[var_name], bins=bins)
        plt.title(var_name, fontproperties=font2)
    plt.tight_layout()
    plt.show()



def hist_var(file_path, var_name,
            bins=100,
            figsize=(8, 6)
            ):
    """
    Histogram of the timeseries data of a single variable within a file.
    """
    try:
        from matplotlib.font_manager import FontProperties
        font1 = FontProperties(fname=r"C:\\Windows\\Fonts\\STFANGSO.ttf", size=14)
        font2 = FontProperties(fname=r"C:\\Windows\\Fonts\\STFANGSO.ttf", size=12)
        font3 = FontProperties(fname=r"C:\\Windows\\Fonts\\STFANGSO.ttf", size=10)
        font4 = FontProperties(fname=r"C:\\Windows\\Fonts\\STFANGSO.ttf", size=7)
    except:
        raise Exception("请确保您的系统为Windows系统，并安装了STFANGSO.ttf字体")

    df=pd.read_excel(file_path)[[var_name]].apply( # 读取指定变量（var_names）的数据
            lambda x:pd.to_numeric(x, errors='coerce') # 对于非数字值，用nan填充
            ).fillna(0) # 对于nan值，用0填充
    plt.figure(figsize=figsize)
    plt.suptitle('Histogram of Variables of {}'.format(
                    os.path.basename(file_path))
                 , fontproperties=font2)
    plt.hist(df[var_name], bins=bins)
    plt.title(var_name, fontproperties=font2)
    plt.tight_layout()
    plt.show()





