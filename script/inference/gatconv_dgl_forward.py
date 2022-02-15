import argparse
import time
import torch
import torch.nn.functional as F
import dgl
# import dgl.data

import torch.nn as nn
import sys
import GPUtil
sys.path.append('../..')
from layers.gatconv_layer import *
from torch.autograd.profiler import profile
from util.indicator import *

import scipy.sparse as sp

def load_dataset(args):
    if args.dataset == 'cora':
        data = dgl.data.CoraGraphDataset()
    elif args.dataset == 'citeseer':
        data = dgl.data.CiteseerGraphDataset()
    elif args.dataset == 'pubmed':
        data = dgl.data.PubmedGraphDataset()
    elif args.dataset == 'reddit':
        data = dgl.data.RedditDataset()
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    g = data[0]
    # add self loop
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    
    
    return g


def main(args):
    g=load_dataset(args)
    g=g.to(args.gpu)
    model=GATConv_dgl(args.in_feats,args.out_feats,args.num_heads).to(args.gpu)
    in_feat=torch.rand(g.num_nodes(),args.in_feats,device=args.gpu)

    if args.profileio:
        profile_start()
        model(g,in_feat)
        profile_end()
        exit()

    maxMemory = 0
    for _ in range(5):
        model(g,in_feat)
        GPUs = GPUtil.getGPUs()
        maxMemory = max(GPUs[args.gpu].memoryUsed, maxMemory)

    
    torch.cuda.synchronize()
    start=time.time()
    
    for _ in range(args.epochs):
        model(g,in_feat)
    torch.cuda.synchronize()
    end=time.time()
    print(maxMemory)
    print("gatconv_dgl forward time:",(end-start)/args.epochs)

    with open('../figure8.csv','a') as f:
        print("gat_with_reorg,latency={}s,memory={}MB".format((end-start)/args.epochs,maxMemory),file=f)

    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument("--in_feats",type=int,default=16)
    parser.add_argument("--out_feats",type=int,default=6)
    parser.add_argument("--dataset",type=str,default="cora")
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=1,
                        help="number of hidden attention heads")
    parser.add_argument('--profileio',type=int,default=0)

    args = parser.parse_args()

    main(args)