"""
本文件定义了一些工具函数，包括训练模型、绘图等，将一些复用次数多的代码封装在函数内，方便调用。
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
import tqdm


class FitHistory:
    '''
    Fit history class to record the training history of a model.
    '''
    def __init__(self):
        self.num_epochs=0
        self.epoch_time=[]
        self.train_loss=[]
        self.train_metric=[]
        self.val_loss=[]
        self.val_metric=[]
        self.metadata=None # 用于保存额外信息

    def update(self, epoch_time, train_loss, train_metric, val_loss, val_metric):
        '''
        Parameters:
        - epoch_time: list. The time of training each epoch.
        - train_loss: list. The loss of training each epoch.
        - train_metric: list. The metric of training each epoch.
        - val_loss: list. The loss of validation each epoch.
        - val_metric: list. The metric of validation each epoch.
        '''
        self.num_epochs+=len(epoch_time)
        self.epoch_time.extend(epoch_time)
        self.train_loss.extend(train_loss)
        self.train_metric.extend(train_metric)
        self.val_loss.extend(val_loss)
        self.val_metric.extend(val_metric)

    def plot(self, figsize=(8,4)):
        plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        plt.plot(self.train_loss, label='train_loss')
        plt.plot(self.val_loss, label='val_loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.train_metric, label='train_metric')
        plt.plot(self.val_metric, label='val_metric')
        plt.xlabel('Epochs')
        plt.ylabel('Metric')
        plt.suptitle("Training History")
        plt.legend()
        plt.tight_layout() # 调整子图间距，防止重叠
        # plt.savefig(save_path, dpi=300, bbox_inches='tight') # 保存图片
        plt.show()
    
    def summary(self):
        print(f'Number of epochs:  {self.num_epochs}')
        print(f'Training time:     {np.sum(self.epoch_time):.4f}s')
        print(f'Training loss:     {self.train_loss[-1]:.4f}')
        print(f'Training metric:   {self.train_metric[-1]:.4f}')
        print(f'Validation loss:   {self.val_loss[-1]:.4f}')
        print(f'Validation metric: {self.val_metric[-1]:.4f}')


def train(MODEL, train_loader, val_loader, optimizer,
            loss_func=nn.MSELoss(),
            metric_func=nn.L1Loss(),
            num_epochs=10,
            device='cpu',
            verbose=1
            ):
    if not hasattr(MODEL, 'label_len'): # 如果模型不含有label_len属性，说明前向传播过程不需要解码器输入
        epoch_time_list=[]
        train_loss_list=[]
        train_metric_list=[]
        val_loss_list=[]
        val_metric_list=[]
        total_time=0.0 # 总训练时间

        for epoch in tqdm.tqdm(range(num_epochs)):
            t1=time.time() # 该轮开始时间
            train_loss, train_metric = 0.0, 0.0 # 本轮的训练loss和metric
            val_loss, val_metric = 0.0, 0.0 # 本轮的验证loss和metric

            # 训练
            MODEL.train() # 切换到训练模式
            for inputs, targets in train_loader: # 分批次遍历训练集
                inputs, targets = inputs.to(device), targets.to(device) # 将数据转移到GPU（如果可用）
                optimizer.zero_grad() # 清空梯度
                outputs = MODEL(inputs)
                loss = loss_func(outputs, targets)
                metric = metric_func(outputs, targets)
                loss.backward() # 反向传播
                optimizer.step() # 更新权重
                train_loss+=loss.item()
                train_metric+=metric.item()

            # 验证
            MODEL.eval() # 切换到验证模式
            with torch.no_grad(): # 关闭梯度计算
                for inputs, targets in val_loader: # 分批次遍历验证集
                    inputs, targets = inputs.to(device), targets.to(device) # 将数据转移到GPU（如果可用）
                    outputs = MODEL(inputs)
                    loss = loss_func(outputs, targets)
                    metric=metric_func(outputs, targets)
                    val_loss+=loss.item()
                    val_metric+=metric.item()

            # 计算各指标的平均值
            average_train_loss=train_loss/len(train_loader) # 本轮的平均训练loss
            average_train_metric=train_metric/len(train_loader) # 本轮的平均训练metric
            average_val_loss=val_loss/len(val_loader) # 本轮的平均验证loss
            average_val_metric=val_metric/len(val_loader) # 本轮的平均验证metric

            # 计算训练用时
            t2=time.time() # 该轮结束时间
            total_time+=(t2-t1) # 累计训练时间

            # 记录本轮各指标值
            epoch_time_list.append(t2-t1)
            train_loss_list.append(average_train_loss)
            train_metric_list.append(average_train_metric)
            val_loss_list.append(average_val_loss)
            val_metric_list.append(average_val_metric)

            # 输出过程信息
            if verbose==1:
                message =f'Epoch [{str(epoch + 1).center(4, " ")}/{num_epochs}], Time: {(t2-t1):.4f}s'
                message+=f', Loss: {average_train_loss:.4f}'
                message+=f', Metric: {average_train_metric:.4f}'
                message+=f', Val Loss: {average_val_loss:.4f}'
                message+=f', Val Metric: {average_val_metric:.4f}'
                print(message)
        print(f'Total Time: {total_time:.4f}s')

        return (epoch_time_list, train_loss_list, train_metric_list, val_loss_list, val_metric_list)

    elif hasattr(MODEL, 'label_len') and MODEL.label_len > 0: # 如果模型含有label_len属性，说明前向传播过程需要解码器输入，训练过程考虑label
        label_len=MODEL.label_len
        output_len=MODEL.output_len
        pred_len=output_len-label_len

        epoch_time_list=[]
        train_loss_list=[]
        train_metric_list=[]
        val_loss_list=[]
        val_metric_list=[]
        total_time=0.0 # 总训练时间

        for epoch in tqdm.tqdm(range(num_epochs)):
            t1 = time.time() # 该轮开始时间
            train_loss, train_metric = 0.0, 0.0 # 本轮的训练loss和metric
            val_loss, val_metric = 0.0, 0.0 # 本轮的验证loss和metric

            # 训练
            MODEL.train() # 切换到训练模式
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device) # 将数据转移到GPU（如果可用）
                optimizer.zero_grad() # 清空梯度
                # decoder input
                dec_inp = torch.zeros_like(targets[:, -pred_len:, :]).float().to(device)
                dec_inp = torch.cat([targets[:, :label_len, :], dec_inp], dim=1).float().to(device)
                # encoder - decoder
                outputs = MODEL(inputs, dec_inp)
                outputs = outputs[:, -pred_len:, :].to(device) # 取待预测时间范围内的数据
                targets = targets[:, -pred_len:, :].to(device) # 取待预测时间范围内的数据
                loss = loss_func(outputs, targets)
                metric = metric_func(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss+=loss.item()
                train_metric+=metric.item()
            
            # 验证
            MODEL.eval() # 切换到验证模式
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device) # 将数据转移到GPU（如果可用）
                    dec_inp = torch.zeros_like(targets[:, -pred_len:, :]).float().to(device)
                    dec_inp = torch.cat([targets[:, :label_len, :], dec_inp], dim=1).float().to(device)
                    outputs = MODEL(inputs, dec_inp)
                    outputs = outputs[:, -pred_len:, :].to(device) # 取待预测时间范围内的数据
                    targets = targets[:, -pred_len:, :].to(device) # 取待预测时间范围内的数据
                    loss = loss_func(outputs, targets)
                    metric = metric_func(outputs, targets)
                    val_loss+=loss.item()
                    val_metric+=metric.item()
            average_train_loss=train_loss/len(train_loader)
            average_train_metric=train_metric/len(train_loader)
            average_val_loss=val_loss/len(val_loader)
            average_val_metric=val_metric/len(val_loader)

            # 计算训练用时
            t2=time.time()
            total_time+=(t2-t1)

            # 记录本轮各指标值
            epoch_time_list.append(t2-t1)
            train_loss_list.append(average_train_loss)
            train_metric_list.append(average_train_metric)
            val_loss_list.append(average_val_loss)
            val_metric_list.append(average_val_metric)
            
            # 输出过程信息
            if verbose==1:
                message =f'Epoch [{str(epoch + 1).center(4, " ")}/{num_epochs}], Time: {(t2-t1):.4f}s'
                message+=f', Loss: {average_train_loss:.4f}'
                message+=f', Metric: {average_train_metric:.4f}'
                message+=f', Val Loss: {average_val_loss:.4f}'
                message+=f', Val Metric: {average_val_metric:.4f}'
                print(message)
        print(f'Total Time: {total_time:.4f}s')

        return (epoch_time_list, train_loss_list, train_metric_list, val_loss_list, val_metric_list)


def get_pred_true_pairs_with_Tensor(model,X,Y):
    '''
    Predict on a given dataset and return the predicted and true values in torch Tensor format.

    :param model: a neural network model (torch.nn.Module)
    :param X: numpy array of shape (n_samples, input_len, n_input_vars)
    :param Y: numpy array of shape (n_samples, output_len, n_output_vars)

    :return: Y_pred: numpy array of shape (n_timesteps, n_output_vars)
    :return: Y_true: numpy array of shape (n_timesteps, n_output_vars)
    '''
    assert type(X)==np.ndarray and X.ndim==3, "X should be a 3D numpy array"
    assert type(Y)==np.ndarray and Y.ndim==3, "Y should be a 3D numpy array"
    assert X.shape[0]==Y.shape[0], "X and Y should have the same number of samples"

    device=next(model.parameters()).device
    X,Y=torch.Tensor(X).to(device),torch.Tensor(Y).to(device)
    n_output_vars=Y.shape[2]
    Y_pred_Tensor=model(X).reshape(-1,n_output_vars) # concatenate the predictions by time order
    Y_true_Tensor=Y.reshape(-1,n_output_vars) # concatenate the true values by time order
    Y_pred, Y_true = Y_pred_Tensor.detach().cpu().numpy(), Y_true_Tensor.detach().cpu().numpy()
    return Y_pred, Y_true # shape: (n_timesteps, n_output_vars)


def get_pred_true_pairs(model,X,Y):
    '''
    Predict on a given dataset and return the predicted and true values in a numpy array.

    :param model: the model to be used
    :param X: numpy array of shape (n_samples, input_len, n_vars)
    :param Y: numpy array of shape (n_samples, output_len, n_vars)

    :return: Y_pred: numpy array of shape (n_timesteps, n_vars)
    :return: Y_true: numpy array of shape (n_timesteps, n_vars)
    '''
    assert type(X)==np.ndarray and X.ndim==3, "X should be a 3D numpy array"
    assert type(Y)==np.ndarray and Y.ndim==3, "Y should be a 3D numpy array"
    assert X.shape[0]==Y.shape[0], "X and Y should have the same number of samples"
    assert X.shape[2]==Y.shape[2], "X and Y should have the same number of variables (channels)"

    n_vars=X.shape[2]
    Y_pred=model(X).reshape(-1,n_vars) # concatenate the predictions by time order
    Y_true=Y.reshape(-1,n_vars) # concatenate the true values by time order
    return Y_pred, Y_true # shape: (n_timesteps, n_vars)


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