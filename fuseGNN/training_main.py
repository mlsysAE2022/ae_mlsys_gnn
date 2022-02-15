"""
Training GCN/GAT on CiteSeer, Cora, PubMed, and Reddit datasets
"""
import argparse
import os
import torch
from fuseGNN.modules import GCN, gcn_config, GAT, gat_config
from fuseGNN.dataloader import Citations, Reddit
from fuseGNN.utils import Logger
import torch_geometric.transforms as T
import torch.nn.functional as F
from tqdm import tqdm
import sys


parser = argparse.ArgumentParser()
# dataset config
parser.add_argument('--data', choices=['CiteSeer', 'Cora', 'PubMed', 'Reddit'], help='dataset name')
parser.add_argument('--model', choices=['GCN', 'GAT'], help='GCN model')
parser.add_argument('--data_path', type=str, default='/raid/datasets/GNN/', help='the path to datasets')
# training config
parser.add_argument('--max_iter', type=int, default=200, help='maximum training iterations')
parser.add_argument('--gpus', type=str, default='0', help='gpu to use')
# logging
parser.add_argument('--log_dir', type=str, default='./log/', help='the path to the logs')
# model configures
parser.add_argument('--mode', choices=['geo', 'ref', 'gas', 'gar'], default='ref', help='run which mode')
parser.add_argument('--flow', choices=['target_to_source', 'source_to_target'], default='target_to_source')

args = parser.parse_args()


# configure CUDA
os.system('export CUDA_VISIBLE_DEVICES=' + args.gpus)
assert torch.cuda.is_available(), 'CUDA is not available'
device = torch.device('cuda')


# configure dataset
path = args.data_path + args.data
try:
    if args.data in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Citations(path, args.data, T.NormalizeFeatures())
    elif args.data in ['Reddit']:
        dataset = Reddit(path)
    data = dataset[0]
    """
    The data contains 
    edge_index=[2, N(e)], test_mask=[N(v)], train_mask=[N(v)], val_mask=[N(v)], x=[N(v), dim], y=[N(v)], deg=[N(v)]
    The deg is very imbalanced, e.g. [1, 168] for Cora, however, I can still try to ignore it at this very begining
    """
except:
    print('The dataset does not exist or is not supported.')
    sys.exit()

data.train_mask = data.train_mask.to(torch.bool)
data.val_mask = data.val_mask.to(torch.bool)
data.test_mask = data.test_mask.to(torch.bool)

if args.model == 'GCN':
    config = gcn_config[args.data]
    model = GCN(num_features=dataset.num_features, hidden=config['hidden'],
                num_classes=dataset.num_classes, cached=True, drop_rate=config['drop_rate'], mode=args.mode, flow=args.flow)
elif args.model == 'GAT':
    config = gat_config[args.data]
    model = GAT(num_features=dataset.num_features, hidden=config['hidden'],
                num_classes=dataset.num_classes, heads=config['head'], drop_rate=config['drop_rate'], mode=args.mode)

logger = Logger(model_=args.model + '_' + args.mode, data_=args.data, log_dir=args.log_dir)

optimizer = torch.optim.Adam([
    dict(params=model.reg_params, weight_decay=config['weight_decay']),
    dict(params=model.non_reg_params, weight_decay=0.)
], lr=config['lr'])

# training
class NodeClassifier:
    def __init__(self, data, model):
        self.data = data.to(device)
        self.model = model.to(device)

    def train(self):
        self.model.train()
        optimizer.zero_grad()
        loss = F.nll_loss(self.model(self.data)[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()

    def test(self):
        self.model.eval()
        logits, accs = self.model(self.data), []
        for _, mask in self.data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(self.data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
        return accs

    def next_fold(self, fold_):
        pass

classifier = NodeClassifier(data, model)

pbar = tqdm(range(args.max_iter))
best_test_acc = 0.
for epoch in pbar:
    config['lr_schedular'].update(epoch, optimizer)
    train_loss = classifier.train()
    train_acc, val_acc, test_acc = classifier.test()
    logger.add_scalar('train_loss', train_loss, epoch)
    logger.add_scalar('train_acc', train_acc, epoch)
    logger.add_scalar('val_acc', val_acc, epoch)
    logger.add_scalar('test_acc', test_acc, epoch)
    logger.log['epoch'].append(epoch)
    if test_acc > best_test_acc:
        best_test_acc = test_acc
    pbar.set_description('Train ACC: %.3f | Test ACC: %.3f | Loss: %.3f' % (train_acc, best_test_acc, train_loss))

logger.write()
"""
summary = torch.cuda.memory_summary(device=device)
print(summary)
mem_stat = torch.cuda.memory_stats(device=device)
print("Peak Memory: %.3f MB" % float(float(mem_stat['allocated_bytes.all.peak']) / 1024. / 1024.))
"""