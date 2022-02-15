"""
    A testbench that verifies training kernels (both forward pass and backward pass)
"""
import argparse
import torch
import torch.nn.functional as F
import torch_scatter
import sys
import torch_geometric.transforms as T
from torch_geometric.utils import add_remaining_self_loops
from fuseGNN.dataloader import Citations
from fuseGNN.convs import geoGCNConv, refGCNConv, garGCNConv, gasGCNConv
from torch_geometric.utils import degree
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/raid/datasets/GNN/')
args = parser.parse_args()

datasets = ['Cora', 'CiteSeer', 'PubMed']
edge_permutes = [True, False]
edge_weights = [True, False]
flows = ['target_to_source', 'source_to_target']

# configure CUDA
assert torch.cuda.is_available(), "CUDA is not available"
device = torch.device('cuda')


class CastratedGCN(torch.nn.Module):
    """
    GCN model for profiling
    """
    def __init__(self, num_features, data=None, hidden=256, num_classes=10, cached=True, fused=False, flow='target_to_source'):
        super(CastratedGCN, self).__init__()
        self.data = data
        if fused:
            self.conv1 = gasGCNConv(in_channels=num_features, out_channels=hidden, cached=cached, 
                                        flow=flow, bias=False)
        else:
            self.conv1 = geoGCNConv(in_channels=num_features, out_channels=hidden, cached=cached,
                                        flow=flow, bias=False)
        
    
    def forward(self, x, data=None, edge_weight=None):
        if data is None:
            data = self.data
        x = self.conv1(x, data.edge_index, edge_weight)
        return x


def single_test(data_name, edge_permute, ew, flow):
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
    if ew:
        # edge_weight = torch.randperm(data.edge_index.size(1)).to(torch.float32).to('cuda') + 1
        edge_weight = torch.abs_(torch.randn(size=(data.edge_index.size(1),), dtype=torch.float32, device=data.edge_index.device)) + 1
        # print(edge_weight.max())
    else:
        edge_weight = None
    if edge_permute:
        r = torch.randperm(data.edge_index.size(1))
        data.edge_index = data.edge_index[:, r]
        if ew:
            edge_weight = edge_weight[r]
    hidden = np.random.randint(low=5, high=1024)
    
    model = CastratedGCN(num_features=dataset.num_features, data=data, hidden=hidden, num_classes=dataset.num_classes, cached=False, fused=False, flow=flow)
    model.to(device)

    fmodel = CastratedGCN(num_features=dataset.num_features, data=data, hidden=hidden, num_classes=dataset.num_classes, cached=False, fused=True, flow=flow)
    fmodel.conv1.dense.weight = torch.nn.Parameter(model.conv1.weight.t())
    # fmodel.conv2.weight = model.conv2.weight
    # fmodel.conv1.dense.bias = torch.nn.Parameter(model.conv1.bias)
    # fmodel.conv1.dense.weight = model.conv1.dense.weight
    # fmodel.conv1.dense.bias = model.conv1.dense.bias
    fmodel.to(device)
    
    model.train()
    fmodel.train()
    
    x_f = data.x.clone().requires_grad_(True)
    x = data.x.clone().requires_grad_(True)
    
    f_res = fmodel(x=x_f, edge_weight=edge_weight)
    ref = model(x=x, edge_weight=edge_weight)
    
    grad = torch.rand_like(f_res)
    
    f_res.backward(grad)
    ref.backward(grad)
    
    grad_x_f = x_f.grad
    grad_x = x.grad
    
    max_error = torch.max((f_res - ref)).item()
    max_error_b = torch.max((grad_x_f - grad_x)).item()
    passed = True
    if max_error > 1e-5 or np.isnan(max_error):
        print("[Forward] on %s, edge_permute %r, edge_weight %r, hidden size %d, flow %s" % (data_name, edge_permute, ew, hidden, flow))
        print("there are %d different entries in overall %d entires. The maximum difference is %f" % 
                    (torch.nonzero(f_res - ref).size(0), f_res.numel(), max_error))
        passed = False
    if max_error_b > 1e-5 or np.isnan(max_error_b):
        print("[Backward] on %s, edge_permute %r, edge_weight %r, hidden size %d, flow %s" % (data_name, edge_permute, ew, hidden, flow))
        print("there are %d different entries in overall %d entires. The maximum difference is %f" % 
                    (torch.nonzero(grad_x_f - grad_x).size(0), grad_x_f.numel(), max_error_b))
        passed = False
    return passed
    

num_exp = 0
num_pass = 0

for ds in datasets:
    for ep in edge_permutes:
        for ew in edge_weights:
            for f in flows:
                for i in range(3):
                    num_exp += 1
                    if single_test(ds, ep, ew, f):
                        num_pass += 1

print("%d out of %d tests passed" % (num_pass, num_exp))