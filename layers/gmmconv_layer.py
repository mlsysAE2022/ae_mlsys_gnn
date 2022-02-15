"""Torch Module for GMM Conv"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn
from torch.nn import init

import sys
sys.path.append('..')
from operators.fused_gmmconv import GmmConvFuse
from operators.fused_gmmconv import GmmConvStash
from operators.fused_gmmconv import GaussianCp
from operators.mhspmm import csrmhspmm

from dgl import function as fn
from dgl.utils import expand_as_pair

class GMMConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 dim,
                 n_kernels,
                 aggregator_type='sum',
                 residual=False,
                 bias=True,
                 allow_zero_in_degree=False):
        super(GMMConv, self).__init__()
        self._out_feats = out_feats
        self._dim = dim
        self._n_kernels = n_kernels
        self._allow_zero_in_degree = allow_zero_in_degree
        if aggregator_type == 'sum':
            self._reducer = fn.sum
        else:
            raise KeyError("Aggregator type {} not recognized.".format(aggregator_type))

        self.mu = nn.Parameter(th.Tensor(n_kernels, dim))
        self.inv_sigma = nn.Parameter(th.Tensor(n_kernels, dim))
        self.fc = nn.Linear(in_feats, n_kernels * out_feats, bias=False)
        if residual:
            if in_feats != out_feats:
                self.res_fc = nn.Linear(in_feats, out_feats, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)

        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        gain = init.calculate_gain('relu')
        init.xavier_normal_(self.fc.weight, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            init.xavier_normal_(self.res_fc.weight, gain=gain)
        init.normal_(self.mu.data, 0, 0.1)
        init.constant_(self.inv_sigma.data, 1)
        if self.bias is not None:
            init.zeros_(self.bias.data)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, rowptr, colind, colptr, rowind, permute, feat, pseudo):
            node_feat = self.fc(feat).view(-1, self._n_kernels, self._out_feats)
            rst = GmmConvFuse(rowptr, colind, colptr, rowind, permute, node_feat, pseudo, self.mu, self.inv_sigma).sum(1)
            # residual connection
            if self.res_fc is not None:
                rst = rst + self.res_fc(feat)
            # bias
            if self.bias is not None:
                rst = rst + self.bias
            return rst


class GMMConv_dgl(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 dim,
                 n_kernels,
                 aggregator_type='sum',
                 residual=False,
                 bias=True,
                 allow_zero_in_degree=False):
        super(GMMConv_dgl, self).__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._dim = dim
        self._n_kernels = n_kernels
        self._allow_zero_in_degree = allow_zero_in_degree
        if aggregator_type == 'sum':
            self._reducer = fn.sum
        else:
            raise KeyError("Aggregator type {} not recognized.".format(aggregator_type))

        self.mu = nn.Parameter(th.Tensor(n_kernels, dim))
        self.inv_sigma = nn.Parameter(th.Tensor(n_kernels, dim))
        self.fc = nn.Linear(self._in_src_feats, n_kernels * out_feats, bias=False)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(self._in_dst_feats, out_feats, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)

        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        gain = init.calculate_gain('relu')
        init.xavier_normal_(self.fc.weight, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            init.xavier_normal_(self.res_fc.weight, gain=gain)
        init.normal_(self.mu.data, 0, 0.1)
        init.constant_(self.inv_sigma.data, 1)
        if self.bias is not None:
            init.zeros_(self.bias.data)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, pseudo):
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, graph)
            graph.srcdata['h'] = self.fc(feat_src).view(-1, self._n_kernels, self._out_feats)
            E = graph.number_of_edges()
            # compute gaussian weight
            gaussian = -0.5 * ((pseudo.view(E, 1, self._dim) -
                                self.mu.view(1, self._n_kernels, self._dim)) ** 2)
            gaussian = gaussian * (self.inv_sigma.view(1, self._n_kernels, self._dim) ** 2)
            gaussian = th.exp(gaussian.sum(dim=-1, keepdim=True)) # (E, K, 1)
            graph.edata['w'] = gaussian
            graph.update_all(fn.u_mul_e('h', 'w', 'm'), self._reducer('m', 'h'))
            rst = graph.dstdata['h'].sum(1)
            # residual connection
            if self.res_fc is not None:
                rst = rst + self.res_fc(feat_dst)
            # bias
            if self.bias is not None:
                rst = rst + self.bias
            return rst

class GMMConv_fuse_no_spmm(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 dim,
                 n_kernels,
                 aggregator_type='sum',
                 residual=False,
                 bias=True,
                 allow_zero_in_degree=False):
        super(GMMConv_fuse_no_spmm, self).__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._dim = dim
        self._n_kernels = n_kernels
        self._allow_zero_in_degree = allow_zero_in_degree
        if aggregator_type == 'sum':
            self._reducer = fn.sum
        else:
            raise KeyError("Aggregator type {} not recognized.".format(aggregator_type))

        self.mu = nn.Parameter(th.Tensor(n_kernels, dim))
        self.inv_sigma = nn.Parameter(th.Tensor(n_kernels, dim))
        self.fc = nn.Linear(self._in_src_feats, n_kernels * out_feats, bias=False)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(self._in_dst_feats, out_feats, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)

        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        gain = init.calculate_gain('relu')
        init.xavier_normal_(self.fc.weight, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            init.xavier_normal_(self.res_fc.weight, gain=gain)
        init.normal_(self.mu.data, 0, 0.1)
        init.constant_(self.inv_sigma.data, 1)
        if self.bias is not None:
            init.zeros_(self.bias.data)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, rowptr, colind, colptr, rowind, permute, feat, pseudo):
        node_feat = self.fc(feat).view(-1, self._n_kernels, self._out_feats)
        # compute gaussian weight
        gaussian = -0.5 * ((pseudo.view(colind.shape[0], 1, self._dim) -
                                self.mu.view(1, self._n_kernels, self._dim)) ** 2)
        gaussian = gaussian * (self.inv_sigma.view(1, self._n_kernels, self._dim) ** 2)
        gaussian = th.exp(gaussian.sum(dim=-1))
        feat_out = csrmhspmm(rowptr, colind, colptr, rowind, permute, node_feat, gaussian)
        rst = feat_out.sum(1)
        # residual connection
        if self.res_fc is not None:
            rst = rst + self.res_fc(feat_dst)
        # bias
        if self.bias is not None:
            rst = rst + self.bias
        return rst



class GMMConv_stash(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 dim,
                 n_kernels,
                 aggregator_type='sum',
                 residual=False,
                 bias=True,
                 allow_zero_in_degree=False):
        super(GMMConv_stash, self).__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._dim = dim
        self._n_kernels = n_kernels
        self._allow_zero_in_degree = allow_zero_in_degree
        if aggregator_type == 'sum':
            self._reducer = fn.sum
        else:
            raise KeyError("Aggregator type {} not recognized.".format(aggregator_type))

        self.mu = nn.Parameter(th.Tensor(n_kernels, dim))
        self.inv_sigma = nn.Parameter(th.Tensor(n_kernels, dim))
        self.fc = nn.Linear(self._in_src_feats, n_kernels * out_feats, bias=False)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(self._in_dst_feats, out_feats, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)

        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        gain = init.calculate_gain('relu')
        init.xavier_normal_(self.fc.weight, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            init.xavier_normal_(self.res_fc.weight, gain=gain)
        init.normal_(self.mu.data, 0, 0.1)
        init.constant_(self.inv_sigma.data, 1)
        if self.bias is not None:
            init.zeros_(self.bias.data)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, rowptr, colind, colptr, rowind, permute, feat, pseudo):
        node_feat = self.fc(feat).view(-1, self._n_kernels, self._out_feats)
        # compute gaussian weight
        rst = GmmConvStash(rowptr, colind, colptr, rowind, permute, node_feat, pseudo, self.mu, self.inv_sigma).sum(1)
        # residual connection
        if self.res_fc is not None:
            rst = rst + self.res_fc(feat_dst)
        # bias
        if self.bias is not None:
            rst = rst + self.bias
        return rst