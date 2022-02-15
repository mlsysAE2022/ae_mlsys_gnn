"""
    A testbench that verifies both forward and backward kernels of GAT
"""
import argparse
import torch
import torch.nn.functional as F
import torch_scatter
import sys
import torch_geometric.transforms as T
from torch_geometric.utils import add_remaining_self_loops
# import gcnlib
from fuseGNN.dataloader import Citations
from fuseGNN.convs import garGATConv, refGATConv, gasGATConv, geoGATConv
from torch_geometric.utils import degree
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/raid/datasets/GNN/')
args = parser.parse_args()

datasets = ['Cora', 'CiteSeer', 'PubMed']
edge_permutes = [True, False]
flows = ['target_to_source', 'source_to_target']

# configure CUDA
assert torch.cuda.is_available(), "CUDA is not available"
device = torch.device('cuda')


class CastratedGAT(torch.nn.Module):
    """
    GCN model for profiling
    """
    def __init__(self, num_features, data=None, hidden=256, heads=8, fused=False, flow='target_to_source'):
        super(CastratedGAT, self).__init__()
        self.data = data
        if fused:
            self.conv1 = garGATConv(in_channels=num_features, out_channels=int(hidden/heads), heads=heads, dropout=0.6, flow=flow)
        else:
            self.conv1 = refGATConv(in_channels=num_features, out_channels=int(hidden/heads), heads=heads, dropout=0.6, flow=flow, return_mask=True)
        
    
    def forward(self, x, data=None, dp_mask=None, dp_mask_self=None):
        if data is None:
            data = self.data
        return self.conv1(x, data.edge_index, dp_mask=dp_mask, dp_mask_self=dp_mask_self)


def single_test(data_name, edge_permute, flow):
    # Configure dataset
    path = args.data_path + data_name
    try:
        dataset = Citations(path, data_name, T.NormalizeFeatures())
        data = dataset[0]
        # todo: preprocess the degree vector
        """
        The data contains 
        edge_index=[2, N(e)], test_mask=[N(v)], train_mask=[N(v)], val_mask=[N(v)], x=[N(v), dim], y=[N(v)], deg=[N(v)]
        The deg is very imbalanced, e.g. [1, 168] for Cora, however, I can still try to ignore it at this very begining
        """
    except:
        print('The dataset does not exist or is not supported.')
        sys.exit()
    data.to(device)
    if edge_permute:
        r = torch.randperm(data.edge_index.size(1))
        data.edge_index = data.edge_index[:, r]
    hidden = np.random.randint(low=1, high=128)
    
    model = CastratedGAT(num_features=dataset.num_features, data=data, hidden=hidden * 8, heads=8, fused=False, flow=flow)
    model.to(device)
    model.train()
    
    fmodel = CastratedGAT(num_features=dataset.num_features, data=data, hidden=hidden * 8, heads=8, fused=True, flow=flow)
    fmodel.to(device)
    fmodel.train()
    
    fmodel.conv1.dense.weight = model.conv1.dense.weight
    fmodel.conv1.dense.bias = model.conv1.dense.bias
    fmodel.conv1.att.data = model.conv1.att.data
    
    x = data.x.clone().requires_grad_(True)
    x_f = data.x.clone().requires_grad_(True)

    # fmodel.conv1.dense.bias = torch.nn.Parameter(model.conv1.bias)
    # fmodel.conv1.dense.weight = model.conv1.dense.weight
    # fmodel.conv1.dense.bias = model.conv1.dense.bias
    # fmodel.to(device)
    ref, dp_mask, dp_mask_self = model(x=x)
    dp_mask = dp_mask.detach()
    dp_mask_self = dp_mask_self.detach()
    f_res = fmodel(x=x_f, dp_mask=dp_mask, dp_mask_self=dp_mask_self)
    grad = torch.rand_like(f_res)
    f_res.backward(grad)
    fgrad = x_f.grad
    ref.backward(grad)
    refgrad = x.grad
    error = torch.abs(f_res - ref)
    graderror = torch.abs(fgrad - refgrad)
    
    fattgrad = fmodel.conv1.att.grad
    refattgrad = model.conv1.att.grad
    attgraderror = torch.abs(fattgrad - refattgrad)
    
    # error = error.ge(1e-5).to(torch.float32)
    # print(error)
    max_error = torch.max(error).item()
    passed = True
    if max_error > 1e-5 or np.isnan(max_error):
        print("[Forward] on %s, edge_permute %r, hidden size %d, flow %s" % (data_name, edge_permute, hidden, flow))
        print("there are %d different entries in overall %d entires. The maximum difference is %f" % 
                    (torch.nonzero(error).size(0), error.numel(), max_error))
        passed = False
    max_graderror = torch.max(graderror).item()
    if max_graderror > 5e-5 or np.isnan(max_graderror):
        print("[Backward] on %s, edge_permute %r, hidden size %d, flow %s" % (data_name, edge_permute, hidden, flow))
        print("there are %d different entries in overall %d entires. The maximum difference is %f" % 
                    (torch.nonzero(graderror).size(0), graderror.numel(), max_graderror))
        passed = False
    max_attgraderror = torch.max(attgraderror).item()
    if max_attgraderror > 1e-5 or np.isnan(max_attgraderror):
        print("[ATT] on %s, edge_permute %r, hidden size %d, flow %s" % (data_name, edge_permute, hidden, flow))
        print("there are %d different entries in overall %d entires. The maximum difference is %f" % 
                    (torch.nonzero(attgraderror).size(0), attgraderror.numel(), max_attgraderror))
        passed = False
    return passed
    

num_exp = 0
num_pass = 0

for ds in datasets:
    for ep in edge_permutes:
        for f in flows:
            for i in range(6):
                num_exp += 1
                if single_test(ds, ep, f):
                    num_pass += 1

print("%d out of %d tests passed" % (num_pass, num_exp))