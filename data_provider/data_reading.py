import numpy as np
import pandas as pd
import pickle

# 专用于已有文件的一个临时封装函数，内部实现随时可能根据数据集格式临时修改
#def load_data(data_path=r'data\data2253.pkl'):
#    """
#    A temporary encapsulation function. Can be customized according to the data set format.
#
#    Args:
#        data_path (str): data file path.
#        
#    Returns:
#        data, var_names, var_units
#            - data: a list of numpy arrays of shape (n_timesteps, n_features), each array is a time series of data,
#            - var_names: a list of variable names,
#            - var_units: a list of variable units.
#    """
#
#    data_list = pickle.load(open(data_path, 'rb'))
#    var_names = [
#        'O2',
#        'CO2',
#        'NOx',
#        'ppm CO',
#        'ppm NO',
#        'ppm NO2',
#        'Smoke Temperature',
#        'Environment Temperature',
#        'Inside Temperature',
#        'Inside Humidity',
#        'Inside Pressure',
#        'Outside Temperature',
#        'Outside Humidity',
#        'Outside Pressure',
#        'FGR',
#        'Fan Frequency',
#        'Load',
#        'Gas Bias Valve',
#        'Fan Bias Valve',
#        'Flame Temperature',
#        'Flame Speed'
#        ]
#    var_units = [
#        '%',
#        '%',
#        'mg/m3',
#        'ppm',
#        'ppm',
#        'ppm',
#        '℃',
#        '℃',
#        '℃',
#        '%',
#        'kPa',
#        '℃',
#        '%',
#        'kPa',
#        '%',
#        '%',
#        '%',
#        '%',
#        '%',
#        '℃',
#        'm/s',
#    ]
#
#    DATA = [data_list[0][:3000,:],
#            data_list[0][3000:6000,:],
#            data_list[0][6000:9000,:],
#            data_list[0][9000:12000,:],
#            data_list[0][12000:15000,:],
#            data_list[0][15000:,:],
#            data_list[1][:3000,:],
#            data_list[1][3000:6000,:],
#            data_list[1][6000:9000,:],
#            data_list[1][9000:,:],
#            data_list[2][:4000,:],
#            data_list[2][4000:8000,:],
#            data_list[2][8000:12000,:],
#            data_list[2][12000:16000,:],
#            data_list[2][16000:20000,:],
#            data_list[2][20000:,:],
#            data_list[3][:4000,:],
#            data_list[3][4000:8000,:],
#            data_list[3][8000:12000,:],
#            data_list[3][12000:16000,:],
#            data_list[3][16000:20000,:],
#            data_list[3][20000:,:],
#            ]
#
#    return DATA, var_names, var_units

# 专用于已有文件的一个临时封装函数，内部实现随时可能根据数据集格式临时修改
def load_data(data_path=r"E:\PythonProjects\gas-data-prediction\data\太原-wavelet-gaussian.xlsx"):
    """
    A temporary encapsulation function. Can be customized according to the dataset format.

    Args:
        data_path (str): data file path.
        
    Returns:
        data, var_names, var_units
            - data: a list of numpy arrays of shape (n_timesteps, n_features), each array is a time series of data,
            - var_names: a list of variable names,
            - var_units: a list of variable units.
    """
    df = pd.read_excel(data_path,sheet_name='Sheet1',header=[0,1])

    var_names = df.columns.get_level_values(0).tolist()
    var_names.remove('时间')
    var_units = df.columns.get_level_values(1).tolist()
    var_units.remove('Unit')
    #print(var_names)
    #print(var_units)
    df.fillna(0, inplace=True)
    #df.info()

    def find_continuous_blocks(df, time_column=('时间','Unit'), threshold='1s'):
        df = df.copy()
        df[time_column] = pd.to_datetime(df[time_column])
    
        time_diff = df[time_column].diff() > pd.Timedelta(threshold) # boolean pd.Series
        break_num = time_diff.cumsum() # int pd.Series
    
        block_indices = []
        for _, group in df.groupby(break_num): # Iterate over sub-DataFrames grouped by the key (break_num)
            # group: pd.DataFrame
            if not group.empty:
                block_indices.append([group.index[0], group.index[-1]])
    
        return block_indices

    block_indices = find_continuous_blocks(df, time_column=('时间','Unit'), threshold='1s')

    DATA=[]
    for (begin,end) in block_indices:
        DATA.append(df[var_names].iloc[begin:end+1].to_numpy(dtype=np.float32))

    return DATA, var_names, var_units



if __name__ == '__main__':
    data, var_names, var_units = load_data()
    print(len(data))
