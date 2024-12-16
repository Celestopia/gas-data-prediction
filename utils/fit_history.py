import numpy as np
import matplotlib.pyplot as plt


class FitHistory:
    '''
    Fit history class to record the training history of a model.

    Example usage:
    ```
    history.update(
            *train(model, train_loader, val_loader, optimizer,
                loss_func=nn.MSELoss(),
                metric_func=nn.L1Loss(),
                num_epochs=num_epochs,
                device=device,
                verbose=1)
            )
    history.plot()
    history.summary()
    ```
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

    def plot(self, figsize=(8,4), save_path=None):
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
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight') # 保存图片
            print(f'Save training history plot to {save_path}')
        plt.show()
    
    def summary(self):
        print(f'Number of epochs:  {self.num_epochs}')
        print(f'Training time:     {np.sum(self.epoch_time):.4f}s')
        print(f'Training loss:     {self.train_loss[-1]:.4f}')
        print(f'Training metric:   {self.train_metric[-1]:.4f}')
        print(f'Validation loss:   {self.val_loss[-1]:.4f}')
        print(f'Validation metric: {self.val_metric[-1]:.4f}')

