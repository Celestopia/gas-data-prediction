import numpy as np
import torch
import torch.nn as nn
import time
import tqdm


def train(model, train_loader, val_loader, optimizer,
            loss_func=nn.MSELoss(),
            metric_func=nn.L1Loss(),
            num_epochs=10,
            device='cpu',
            verbose=1
            ):
    """
    Train a model using the given data loaders and settings.

    Args:
        model: The pytorch model to be trained.
        train_loader: The data loader for the training set.
        val_loader: The data loader for the validation set.
        optimizer: The optimizer to be used for training.
        loss_func: The loss function to be used for training.
        metric_func: The metric function to be used for evaluation.
        num_epochs: The **maximum number** of epochs to train. Note that early stopping may be triggered.
        device: The device to be used for training. options: ['cpu', 'cuda'].
        verbose: The level of verbosity. options: [0, 1]. 0: no output, 1: output each epoch.

    Returns:
        (epoch_time_list, train_loss_list, train_metric_list, val_loss_list, val_metric_list) (tuple of list):
        - A tuple of lists containing the training time, training loss, training metric, validation loss, and validation metric for each epoch.
    """
    epoch_time_list=[]
    train_loss_list=[]
    train_metric_list=[]
    val_loss_list=[]
    val_metric_list=[]
    total_time=0.0 # Total training time

    for epoch in tqdm.tqdm(range(num_epochs)):
        t1=time.time() # Start time of this epoch
        train_loss, train_metric = 0.0, 0.0
        val_loss, val_metric = 0.0, 0.0

        # Train
        model.train() # Switch to training mode
        for inputs, targets in train_loader: # Traverse the training set in batches
            inputs, targets = inputs.to(device), targets.to(device) # Move data to GPU (if available)
            optimizer.zero_grad() # Empty the gradient
            outputs = model(inputs)
            loss = loss_func(outputs, targets)
            metric = metric_func(outputs, targets)
            loss.backward() # Calculate gradients
            optimizer.step() # Update model parameters
            train_loss+=loss.item()
            train_metric+=metric.item()

        # Validate
        model.eval() # Switch to evaluation mode
        with torch.no_grad():
            for inputs, targets in val_loader: # Traverse the validation set in batches
                inputs, targets = inputs.to(device), targets.to(device) # Move data to GPU (if available)
                outputs = model(inputs)
                loss = loss_func(outputs, targets)
                metric=metric_func(outputs, targets)
                val_loss+=loss.item()
                val_metric+=metric.item()

        # Calculate the averaged value of each metric and loss
        average_train_loss=train_loss/len(train_loader) # average training loss of this epoch
        average_train_metric=train_metric/len(train_loader) # average training metric of this epoch
        average_val_loss=val_loss/len(val_loader) # average validation loss of this epoch
        average_val_metric=val_metric/len(val_loader) # average validation metric of this epoch

        # Calculate time
        t2=time.time() # Ending time of this epoch
        total_time+=(t2-t1) # Training time of this epoch

        # Record each value
        epoch_time_list.append(t2-t1)
        train_loss_list.append(average_train_loss)
        train_metric_list.append(average_train_metric)
        val_loss_list.append(average_val_loss)
        val_metric_list.append(average_val_metric)

        # Output epoch information
        if verbose==1:
            message =f'Epoch [{str(epoch + 1).center(4, " ")}/{num_epochs}], Time: {(t2-t1):.4f}s'
            message+=f', Loss: {average_train_loss:.4f}'
            message+=f', Metric: {average_train_metric:.4f}'
            message+=f', Val Loss: {average_val_loss:.4f}'
            message+=f', Val Metric: {average_val_metric:.4f}'
            print(message)
        
        # Set early stopping rule
        if epoch>30 and epoch%10==0:
            average_val_loss=np.mean(val_loss_list[-20:])
            average_val_loss_prev=np.mean(val_loss_list[-30:-10])
            if average_val_loss>0.95*average_val_loss_prev:
                print("Early stopping at epoch {}.".format(epoch+1))
                break
    
    if verbose==1:
        print(f'Total Time: {total_time:.4f}s')

    return (epoch_time_list, train_loss_list, train_metric_list, val_loss_list, val_metric_list)