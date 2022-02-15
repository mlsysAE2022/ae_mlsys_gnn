import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import KNNGraph
import sys
import GPUtil
sys.path.append('../..')
from layers.edgeconv_layer import EdgeConv_reorg
import time
import numpy as np
from util.indicator import *


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dgl.data.utils import download, get_download_dir

from functools import partial
import tqdm
import urllib
import os
import argparse

from torch.autograd.profiler import profile

parser = argparse.ArgumentParser()
parser.add_argument('--dataset-path', type=str, default='')
parser.add_argument('--load-model-path', type=str, default='')
parser.add_argument('--save-model-path', type=str, default='')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--num-workers', type=int, default=0)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--k',type=int,default=40)
parser.add_argument('--out-feat',type=int,default=64)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--profileio',type=int,default=0)

args = parser.parse_args()



dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data=torch.randn(args.batch_size*1024,3)
nng = KNNGraph(args.k)
model=EdgeConv_reorg(3,args.out_feat).to(dev)
g = nng(data).to(dev)
data=data.to(dev)

maxMemory = 0
if not args.profileio:
    for _ in range(5):
        model(g,data)
        GPUs = GPUtil.getGPUs()
        maxMemory = max(GPUs[args.gpu].memoryUsed, maxMemory)
else:
    profile_start()
    model(g,data)
    profile_end()
    exit()

torch.cuda.synchronize()
start=time.time()

for _ in range(args.epochs):
    model(g,data)
    

torch.cuda.synchronize()
end=time.time()
print(maxMemory)
print("edgeconv_reorg forward time:",(end-start)/args.epochs)

with open('../figure9.csv','a') as f:
    print("edgeconv_without_fusion,latency={}s,memory={}MB".format((end-start)/args.epochs,maxMemory),file=f)