import numpy as np
import torch
import torch.nn as nn
import time
import tqdm


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
            
            # 设置提前停止规则
            if epoch>30 and epoch%10==0:
                average_val_loss=np.mean(val_loss_list[-20:])
                average_val_loss_prev=np.mean(val_loss_list[-30:-10])
                if average_val_loss>0.95*average_val_loss_prev:
                    print("Early stopping at epoch {}.".format(epoch+1))
                    break

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
            
            # 设置提前停止规则
            if epoch>30 and epoch%10==0:
                average_val_loss=np.mean(val_loss_list[-20:])
                average_val_loss_prev=np.mean(val_loss_list[-30:-10])
                if average_val_loss>0.95*average_val_loss_prev:
                    print("Early stopping at epoch {}.".format(epoch+1))
                    break

        print(f'Total Time: {total_time:.4f}s')

        return (epoch_time_list, train_loss_list, train_metric_list, val_loss_list, val_metric_list)