import json
import os
from tensorboardX import SummaryWriter


class Logger:
    """
    Record the results and store them to the json file
    """
    def __init__(self, model_, data_, log_dir):
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        self.log = {
            'model': model_,
            'data': data_,
            'train_loss': [],
            'train_acc': [],
            'val_acc': [],
            'test_acc': [],
            'epoch': [],
        }
        self.exp_name = model_ + '_' + data_
        json_dir = log_dir + self.exp_name
        self.writer = SummaryWriter(log_dir=json_dir)
        if not os.path.exists(json_dir):
            os.mkdir(json_dir)
        self.log_file = json_dir + 'log.json'

    def add_scalar(self, key, value, epoch):
        self.log[key].append(value)
        self.writer.add_scalar(tag=key, scalar_value=value, global_step=epoch)

    def write(self):
        with open(self.log_file, 'w') as file:
            json.dump(self.log, file)
        self.writer.close()
        