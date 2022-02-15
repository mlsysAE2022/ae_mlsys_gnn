import argparse
import time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
import scipy._build_utils.system_info
import sys
import GPUtil
sys.path.append('../..')
from util.indicator import *
from layers.gmmconv_layer import GMMConv_dgl

def main(args):
    # load and preprocess dataset
    data = load_data(args)
    g = data[0]
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        g = g.to(args.gpu)
    n_nodes = g.number_of_nodes()
    features = torch.rand(n_nodes, args.in_feat).to(args.gpu)
    in_feats = features.shape[1]

    # graph preprocess and calculate normalization factor
    g = g.remove_self_loop().add_self_loop()
    n_edges = g.number_of_edges()
    pseudo = torch.rand(n_edges, args.pseudo_dim).to(args.gpu)

    model = GMMConv_dgl(args.in_feat, args.out_feat, args.pseudo_dim, args.n_kernels)

    if cuda:
        model.cuda()
    
    if(args.profileio):
        profile_start()
        model(g, features, pseudo)
        profile_end()
        exit()

    # warmup 
    maxMemory = 0
    for _ in range(5):
        model(g, features, pseudo)
        GPUs = GPUtil.getGPUs()
        maxMemory = max(GPUs[args.gpu].memoryUsed, maxMemory)

    torch.cuda.synchronize()
    start=time.time()

    for epoch in range(args.n_epochs):
        model(g, features, pseudo)

    torch.cuda.synchronize()
    end=time.time()

    print(maxMemory)
    print("gmmconv forward time:", (end-start)/args.n_epochs)

    with open('../figure9.csv','a') as f:
        print("monet_without_fusion,latency={}s,memory={}MB".format((end-start)/args.n_epochs,maxMemory),file=f)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MoNet on citation network')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--in-feat", type=int, default=16,
                        help="size of in feature")
    parser.add_argument("--out-feat", type=int, default=16,
                        help="size of out feature")
    parser.add_argument("--pseudo-dim", type=int, default=2,
                        help="Pseudo coordinate dimensions in GMMConv, 2 for cora and 3 for pubmed")
    parser.add_argument("--n-kernels", type=int, default=3,
                        help="Number of kernels in GMMConv layer")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--profileio", type=int, default=0,
                        help="1 for profile io")
    args = parser.parse_args()
    print(args)

    main(args)