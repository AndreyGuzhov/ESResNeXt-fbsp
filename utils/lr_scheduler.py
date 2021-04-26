import torch


class WarmUpStepLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 cold_epochs: int,
                 warm_epochs: int,
                 step_size: int,
                 gamma: float = 0.1,
                 last_epoch: int = -1):

        self.cold_epochs = cold_epochs
        self.warm_epochs = warm_epochs
        self.step_size = step_size
        self.gamma = gamma

        super(WarmUpStepLR, self).__init__(optimizer=optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch < self.cold_epochs:
            return [base_lr * 0.1 for base_lr in self.base_lrs]
        elif self.last_epoch < self.cold_epochs + self.warm_epochs:
            return [
                base_lr * 0.1 + (1 + self.last_epoch - self.cold_epochs) * 0.9 * base_lr / self.warm_epochs
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr * self.gamma ** ((self.last_epoch - self.cold_epochs - self.warm_epochs) // self.step_size)
                for base_lr in self.base_lrs
            ]


class WarmUpExponentialLR(WarmUpStepLR):

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 cold_epochs: int,
                 warm_epochs: int,
                 gamma: float = 0.1,
                 last_epoch: int = -1):

        self.cold_epochs = cold_epochs
        self.warm_epochs = warm_epochs
        self.step_size = 1
        self.gamma = gamma

        super(WarmUpStepLR, self).__init__(optimizer=optimizer, last_epoch=last_epoch)