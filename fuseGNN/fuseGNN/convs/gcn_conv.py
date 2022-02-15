import torch
import torch_scatter
from torch_geometric.nn import GCNConv as geoGCNConv
from fuseGNN.functional import coo2csr, gcn_gar_edge_weight, csr2csc, fused_gar_agg , gcn_gas_edge_weight, fused_gas_agg
import torch.nn.functional as F


# the reference module


class refGCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, cached=False, bias=True, 
                 flow='target_to_source'):
        """
        Args:
            in_channels (int): Size of each input sample
            out_channels (int): Size of each output sample.
            cached (bool, optional): if set to True, the layer will cache
                the computation of D^{-0.5}\hat{A}of D^{-0.5} on the first
                execution, and it will be used for further executions.
                This is only helpful in transductive learning
            bias (bool, optional): If set to True, there will be a bias
            flow (str): could be the following two conditions
                'source_to_target': edge_index[0] is the source nodes, [1] is the target nodes
                'target_to_source': edge_index[0] is the target nodes, [0] is the source nodes
            fused (bool, optional): If set to True, the gcnlib.gcn_aggregate_f_cuda will be used. Default: False,
            verify (bool, optional): If set to True, it will output the difference between fused and unfused version
        """
        super(refGCNConv, self).__init__()
        self.flow = flow
        assert self.flow in ['source_to_target', 'target_to_source']
        self.tid, self.sid = (0, 1) if self.flow == 'target_to_source' else (1, 0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        # The dense layer for Update stage
        self.dense = torch.nn.Linear(in_features=in_channels, out_features=out_channels, bias=bias)
        
        self.cached = cached
        self.cached_num_edges = None
        self.cached_deg = None
        self.cached_deg_int = None
        self.cached_weight = None
        self.cached_edge_ptr = None
    
    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x (float32 [N(v), in_channels]) : matrix of feature vectors
            edge_index (int64 [2, N(v)]) : the list of edges
            edge_weight (float32 [N(v)], optional) : the weight on each edge, default: 1
        """
        # step 0: split the edge index and convert them to int32
        src_index, tar_index = (edge_index[self.sid], edge_index[self.tid])
        
        # step 1: Update, forward input features into linear layer
        x = self.dense(x)
        
        # step 2: getting the D^{-0.5}\hat{A}of D^{-0.5} matrix
        self.get_adj(src_index, tar_index, x.size(0), edge_weight)
        
        # Aggregation stage
        return self.propagate(x, src_index, tar_index)
        
    def get_adj(self, src_index, tar_index, num_nodes, edge_weight=None):
        """
        Args:
            src_index (int64 [N(e)]) : COO index of source
            tar_index (int64 [N(e)]) : COO index of target
            num_nodes (int64): number of nodes in the input graph
            edge_weight (float32 [N(e)], optional) : the weight on each edge, default: 1
        """
        if self.cached and self.cached_num_edges is not None:  
            # when the result is cached, and this is not the first execution
            if src_index.size(0) != self.cached_num_edges:
                raise RuntimeError(
                    'Chached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the cached=True argument in its consturctor'.format(
                        self.cached_num_edges, src_index.size(0)))
        
        if not self.cached or self.cached_num_edges is None:
            # when the result is not cached, or its the first execution
            self.cached_num_edges = src_index.size(0)
            # update the edge weight based on degree information
            self.processing_edge(src_index, tar_index, num_nodes, edge_weight)
    
    def processing_edge(self, src_index, tar_index, num_nodes, edge_weight=None):
        """
        Update the edge_weights with degree information
        w = w/\sqrt((d_s + 1)(d_t + 1))
        Remark: we don't add the self-loops into the edges, instead we will add them manually.
        Args:
            src_index (int64 [N(e)]) : COO index of source
            tar_index (int64 [N(e)]) : COO index of target
            num_nodes (int64): number of nodes in the input graph
            edge_weight (float32 [N(v)], optional) : the weight on each edge, default: 1
        """
        if edge_weight is None:
            edge_weight = torch.ones(size=(src_index.size(0),), dtype=torch.float32, device=src_index.device)
        
        # The index to get degree is the first vector of edge_index
        if self.flow == "source_to_target":
            deg = torch_scatter.scatter_add(src=edge_weight, index=src_index, dim=0, dim_size=num_nodes) + 1
        else:
            deg = torch_scatter.scatter_add(src=edge_weight, index=tar_index, dim=0, dim_size=num_nodes) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        edge_weight = deg_inv_sqrt[src_index] * edge_weight * deg_inv_sqrt[tar_index]
        self.cached_deg = deg
        self.cached_weight = edge_weight
        self.cached_num_edges = src_index.size(0)
        
    def propagate(self, feature, src_index, tar_index):
        """
        Args:
            feature: the feature vectors float32 [N(v), dim]
            edge_index: the list of edges [2, N(e)]
        """
        # step 1: scatter the feature vectors to the extended feature map for reduction
        extended_fm = torch.index_select(feature, dim=0, index=src_index)
        
        # step 2: apply the weights
        extended_fm = extended_fm * self.cached_weight.unsqueeze(1)
        
        # step 3: scatter to the output feature
        out = torch_scatter.scatter_add(src=extended_fm, index=tar_index, dim=0, dim_size=feature.size(0))
        
        # step 4: apply self-loops
        out += feature * self.cached_deg.pow(-1).unsqueeze(1)
        
        return out



# the final fused version

class garGCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, cached=False, bias=True, 
                 flow='target_to_source'):
        """
        Args:
            in_channels (int): Size of each input sample
            out_channels (int): Size of each output sample.
            cached (bool, optional): if set to True, the layer will cache
                the computation of D^{-0.5}\hat{A}of D^{-0.5} on the first
                execution, and it will be used for further executions.
                This is only helpful in transductive learning
            bias (bool, optional): If set to True, there will be a bias
            flow (str): could be the following two conditions
                'source_to_target': edge_index[0] is the source nodes, [1] is the target nodes
                'target_to_source': edge_index[0] is the target nodes, [0] is the source nodes
        """
        super(garGCNConv, self).__init__()
        self.flow = flow
        assert self.flow in ['source_to_target', 'target_to_source']
        
        self.tid, self.sid = (0, 1) if self.flow == 'target_to_source' else (1, 0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        # The dense layer for Update stage
        self.dense = torch.nn.Linear(in_features=in_channels, out_features=out_channels, bias=bias)
        
        self.cached = cached
        
        self.cached_tar_ptr = None
        self.cached_src_index = None
        self.cached_edge_weight_f = None
        self.cached_self_edge_weight = None
        self.cached_src_ptr = None
        self.cached_tar_index = None
        self.cached_edge_weight_b = None
        
        self.cached_num_edges = None

        # It seems that using multiple streams doesn't help
        # self.stream1 = torch.cuda.Stream()
        # self.stream2 = torch.cuda.Stream()
    
    def forward(self, x, edge_index=None, edge_weight=None, self_edge_weight=None, tar_ptr=None, 
                src_index=None, src_ptr=None, tar_index=None, edge_weight_b=None):
        
        x = self.dense(x)
        if not self.cached or self.cached_num_edges is None:  
            # when the results are not cached, or it is the first execution.
            if tar_ptr is not None: # when the CSR & CSC format are provided
                self.cached_tar_ptr = tar_ptr
                self.cached_src_index = src_index
                self.cached_edge_weight_f = edge_weight
                self.cached_self_edge_weight = self_edge_weight
                self.cached_src_ptr = src_ptr
                self.cached_tar_index = tar_index
                self.cached_edge_weight_b = edge_weight_b
                
                self.cached_num_edges = tar_ptr.size(0)
            else:
                num_nodes = x.size(0)
                # convert the edge lists to int32
                edge_index = edge_index.to(torch.int32)
                src_index, tar_index = (edge_index[self.sid], edge_index[self.tid])
                # convert coo format to csr format
                self.cached_src_index, tar_index, self.cached_tar_ptr, edge_weight_f = coo2csr(src_index, tar_index, 
                                                                       num_nodes, edge_weight, False)
                # update edge weight
                self.cached_edge_weight_f, self.cached_self_edge_weight = gcn_gar_edge_weight(self.cached_src_index, 
                                                                                              self.cached_tar_ptr, tar_index,
                                                                                              num_nodes, edge_weight_f,
                                                                                              self.flow)
                # get the csc format for backward pass
                self.cached_src_ptr, self.cached_tar_index, self.cached_edge_weight_b = csr2csc(self.cached_tar_ptr,
                                                                                                self.cached_src_index,
                                                                                                self.cached_edge_weight_f,
                                                                                                num_nodes)
                self.cached_num_edges = self.cached_tar_ptr.size(0)
            
        return fused_gar_agg(feature=x, src_index=self.cached_src_index, tar_ptr=self.cached_tar_ptr,
                             edge_weight_f=self.cached_edge_weight_f, self_edge_weight=self.cached_self_edge_weight,
                             tar_index=self.cached_tar_index, src_ptr=self.cached_src_ptr,
                             edge_weight_b=self.cached_edge_weight_b, require_edge_weight=False)


# the GAS version

class gasGCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, cached=False, bias=True, 
                 flow='target_to_source'):
        """
        Args:
            in_channels (int): Size of each input sample
            out_channels (int): Size of each output sample.
            cached (bool, optional): if set to True, the layer will cache
                the computation of D^{-0.5}\hat{A}of D^{-0.5} on the first
                execution, and it will be used for further executions.
                This is only helpful in transductive learning
            bias (bool, optional): If set to True, there will be a bias
            flow (str): could be the following two conditions
                'source_to_target': edge_index[0] is the source nodes, [1] is the target nodes
                'target_to_source': edge_index[0] is the target nodes, [0] is the source nodes
        """
        super(gasGCNConv, self).__init__()
        self.flow = flow
        assert self.flow in ['source_to_target', 'target_to_source']
        
        self.tid, self.sid = (0, 1) if self.flow == 'target_to_source' else (1, 0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        # The dense layer for Update stage
        self.dense = torch.nn.Linear(in_features=in_channels, out_features=out_channels, bias=bias)
        
        self.cached = cached
        self.cached_num_edges = None
        
        self.cached_src_index = None
        self.cached_tar_index = None
        self.cached_edge_weight = None
        self.cached_self_edge_weight = None
        # It seems that using multiple streams doesn't help
        # self.stream1 = torch.cuda.Stream()
        # self.stream2 = torch.cuda.Stream()
    
    def forward(self, x, edge_index=None, edge_weight=None, src_index=None, tar_index=None, self_edge_weight=None):
        x = self.dense(x)
        if not self.cached or self.cached_num_edges is None:
            if self_edge_weight is not None:
                self.cached_src_index = src_index
                self.cached_tar_index = tar_index
                self.cached_edge_weight = edge_weight
                self.cached_self_edge_weight = self_edge_weight
                self.cached_num_edges = self.cached_src_index.size(0)
            else:
                num_nodes = x.size(0)
                edge_index = edge_index.to(torch.int32)
                self.cached_src_index, self.cached_tar_index = (edge_index[self.sid], edge_index[self.tid])
                self.cached_edge_weight, self.cached_self_edge_weight = gcn_gas_edge_weight(self.cached_src_index,
                                                                                            self.cached_tar_index,
                                                                                            num_nodes,
                                                                                            edge_weight,
                                                                                            self.flow)
                self.cached_num_edges = self.cached_src_index.size(0)
        return fused_gas_agg(feature=x, src_index=self.cached_src_index, tar_index=self.cached_tar_index,
                             edge_weight=self.cached_edge_weight, self_edge_weight=self.cached_self_edge_weight,
                             require_edge_weight=False)
