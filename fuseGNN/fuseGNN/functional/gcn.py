import torch
import fgnn_gcn

# fused get edge weight function for GAR model

class Inv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, degree):
        self_edge_weight = 1./degree
        return self_edge_weight
    @staticmethod
    def backward(ctx, gred_self_edge_weight):
        return None

class GCNGAREdgeWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src_index, tar_ptr, tar_index, num_nodes, edge_weight, flow):
        edge_weight, degree = fgnn_gcn.gcn_gar_edge_weight(src_index, tar_ptr, tar_index, num_nodes, edge_weight,
                                                             flow=='target_to_source')
        return edge_weight, degree
    
    @staticmethod
    def backward(ctx, grad_edge_weight, grad_degree):
        return None, None, None, None, None, None

def gcn_gar_edge_weight(src_index, tar_ptr, tar_index, num_nodes, edge_weight, flow):
    edge_weight, degree = GCNGAREdgeWeight.apply(src_index, tar_ptr, tar_index, num_nodes, edge_weight, flow)
    self_edge_weight = Inv.apply(degree)
    return edge_weight, self_edge_weight
    
# fused get edge weight function for GAS model

class GCNGASEdgeWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src_index, tar_index, num_nodes, edge_weight, flow):
        weight_to_cache, degree_to_cache = fgnn_gcn.gcn_gas_edge_weight(src_index, tar_index, num_nodes, edge_weight,
                                                                          flow=='target_to_source')
        return weight_to_cache, degree_to_cache
    
    @staticmethod
    def backward(ctx, grad_edge_weight, grad_degree):
        return None, None, None, None, None, None
    
def gcn_gas_edge_weight(src_index, tar_index, num_nodes, edge_weight, flow):
    edge_weight, degree = GCNGASEdgeWeight.apply(src_index, tar_index, num_nodes, edge_weight, flow)
    self_edge_weight = Inv.apply(degree)
    return edge_weight, self_edge_weight
