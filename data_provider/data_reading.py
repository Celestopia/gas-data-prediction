import numpy as np
import pandas as pd
import pickle

# 专用于已有文件的一个临时封装函数，内部实现随时可能根据数据集格式临时修改
def load_data(data_path=r'feature_reconstruction\data1344.pkl'):
    """
    A temporary encapsulation function.
    Can be customized according to the data set format.

    :param data_path: data file path
    :return data, var_names, var_units:
    
    - data: a list of numpy arrays of shape (n_timesteps, n_features), each array is a time series of data,
    - var_names: a list of variable names,
    - var_units: a list of variable units.
    """

    data_list = pickle.load(open(data_path, 'rb'))
    var_names = [
        'O2',
        'CO2',
        'NOx',
        'ppm CO',
        'ppm NO',
        'ppm NO2',
        'Smoke Temperature',
        'Environment Temperature',
        'Pumping Flow',
        'Inside Temperature',
        'Inside Humidity',
        'Inside Pressure',
        'Outside Temperature',
        'Outside Humidity',
        'Outside Pressure',
        'FGR',
        'Fan Frequency',
        'Load',
        'Gas Bias Valve',
        'Fan Bias Valve',
        'Flame Temperature',
        'Flame Speed'
        ]
    var_units = [
        '%',
        '%',
        'mg/m3',
        'ppm',
        'ppm',
        'ppm',
        '℃',
        '℃',
        'L/min',
        '℃',
        '%',
        'kPa',
        '℃',
        '%',
        'kPa',
        '%',
        '%',
        '%',
        '%',
        '%',
        '℃',
        'm/s',
    ]

    DATA = [data_list[0][:1000,:],
            data_list[0][1000:2000,:],
            data_list[0][2000:,:],
            data_list[1][:1000,:],
            data_list[1][1000:2000,:],
            data_list[1][2000:,:],
            data_list[2][:1500,:],
            data_list[2][1500:3000,:],
            data_list[2][3000:,:],
            data_list[3][:2000,:],
            data_list[3][2000:4000,:],
            data_list[3][4000:,:],
            ]

    return DATA, var_names, var_units