class LrSchedular:
    def __init__(self, init_lr, mode, **kwargs):
        self.lr = init_lr
        self.mode = mode
        assert self.mode in ['constant', 'step_decay'], "Only 'constant' and 'step_decay' are supported"
        if mode == 'step_decay':
            self.interval = kwargs['interval']
            self.rate = kwargs['rate']

    def update(self, epoch, optimizer):
        if self.mode == 'step_decay':
            if epoch % self.interval == 0:
                self.lr = self.lr * self.rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.lr
