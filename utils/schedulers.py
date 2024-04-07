import math

from torch.optim import Optimizer


class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class CosineSchedule(_LRScheduler):

    def __init__(self, optimizer, K, warmup_epochs=3):
        self.K = K
        self.warmup_epochs = min(warmup_epochs, max(K // 5, 1))
        super().__init__(optimizer, -1)

    def cosine(self, base_lr):
        if self.last_epoch < self.warmup_epochs:
            return base_lr * (self.last_epoch + 1) / (self.warmup_epochs + 1)
        adjusted_epoch = self.last_epoch - self.warmup_epochs
        adjusted_total = max(self.K - 1 - self.warmup_epochs, 1)
        return base_lr * (1 + math.cos(math.pi * adjusted_epoch / adjusted_total)) / 2

    def get_lr(self):
        return [self.cosine(base_lr) for base_lr in self.base_lrs]
