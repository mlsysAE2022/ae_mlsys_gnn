import torch
import torch.nn as nn
import dgl.function as fn
import sys
import argparse
import numpy as np
import time
import torch.nn.functional as F
import dgl
sys.path.append('../..')
from layers.gatconv_layer import GATConv_dgl as GATConv
from tqdm import tqdm
from util.indicator import *
from dgl.data import register_data_args
from torch.autograd.profiler import profile

import torch.nn as nn
import GPUtil

class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],negative_slope))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],negative_slope))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1], negative_slope))

    # @profile_every(1)
    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        return accuracy(logits, labels)


def main(args):
    # load and preprocess dataset
    if args.dataset == 'cora':
        data = dgl.data.CoraGraphDataset()
    elif args.dataset == 'citeseer':
        data = dgl.data.CiteseerGraphDataset()
    elif args.dataset == 'pubmed':
        data = dgl.data.PubmedGraphDataset()
    elif args.dataset == 'reddit':
        data = dgl.data.RedditDataset()
    elif args.dataset == 'ogbn-proteins' or args.dataset == 'ogbn-mag':
        from ogb.nodeproppred import DglNodePropPredDataset
        dataset = DglNodePropPredDataset(name='ogbn-proteins')
        g, labels = dataset[0]
        split_idx = dataset.get_idx_split()
        train_label = dataset.labels[split_idx['train']]
        valid_label = dataset.labels[split_idx['valid']]
        test_label = dataset.labels[split_idx['test']]
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    g = data[0]
    # add self loop
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
     
    g=g.int().to(args.gpu) 
    features = g.ndata['feat'].to(args.gpu)
    labels = g.ndata['label'].to(args.gpu)    
    train_mask = g.ndata['train_mask'].to(args.gpu)
    val_mask = g.ndata['val_mask'].to(args.gpu)
    test_mask = g.ndata['test_mask'].to(args.gpu)
    num_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = g.number_of_edges()

    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d
      #Input features %d
     """ %
          (n_edges, n_classes,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
           test_mask.int().sum().item(),features.shape[1]))
    
    # create model
    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
    model = GAT(g,
                args.num_layers,
                num_feats,
                args.num_hidden,
                n_classes,
                heads,
                F.elu,
                args.in_drop,
                args.attn_drop,
                args.negative_slope,
                args.residual)

    print("in_feat:",num_feats)
    print("num_layers:",args.num_layers+1)
    print('hidden size:',args.num_hidden)
    print('out_feat:',n_classes)

    model.to(args.gpu)
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    
    # initialize graph
    dur = []


    maxMemory = 0
    for _ in range(10):
        model.train()
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        GPUs = GPUtil.getGPUs()
        maxMemory = max(GPUs[args.gpu].memoryUsed, maxMemory)  

    print('profile training')
    torch.cuda.synchronize()
    start=time.time()
    for epoch in tqdm(range(args.epochs)):
        model.train()
        if(args.profileio):
            profile_start()
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if(args.profileio):
            profile_end()
            exit()
        # print(loss.item())       
    torch.cuda.synchronize()
    end=time.time()
    train_time=(end-start)/args.epochs

    acc = evaluate(model, features, labels, test_mask)
    print("Test Accuracy {:.4f}".format(acc))

    
    # print('profile inference')
    # torch.cuda.synchronize()
    # start=time.time()
    # for epoch in range(args.epochs):
    #     model.eval()
    #     logits = model(features)
    #     loss = loss_fcn(logits[train_mask], labels[train_mask])
        
    # torch.cuda.synchronize()
    # end=time.time()
    # inference_time=(end-start)/args.epochs


    print("train time=",train_time)

    # with LineProfiler():
    #     model.train()
    #     logits = model(features)
    #     loss = loss_fcn(logits[train_mask], labels[train_mask])
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    with open("../{}".format(args.output),'a') as f:
        print("train_GAT_dgl,dataset={},head={},f={},latency={}s,memory={}MB".format(args.dataset,args.num_heads,args.num_hidden,train_time,maxMemory),file=f) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument("--dataset",type=str,default="cora")
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=8,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=0,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=0,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--fastmode', action="store_true", default=False,
                        help="skip re-evaluate the validation set")
    parser.add_argument("--profileio", type=int, default=0,
                    help="1 for profile io")
    parser.add_argument("--output",type=str,default="figure7.csv")
    args = parser.parse_args()

    main(args)