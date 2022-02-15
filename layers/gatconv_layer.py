from torch import nn
import sys

import torch
from torch._C import device
sys.path.append('..')
from operators.fused_gat import fused_gat_op,fused_gat_stash_op,fused_gat_fusescatter


class GATConv(nn.Module): # our gat layer
    def __init__(self,
                in_feats,
                out_feats,
                num_heads,
                negative_slope=0.2,

                ):
        super(GATConv,self).__init__()
        self.in_feats=in_feats
        self.out_feats=out_feats
        self.num_heads=num_heads
        self.W = nn.Parameter(torch.FloatTensor(in_feats, out_feats * num_heads))
        self.attn_l = nn.Parameter(torch.zeros(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.zeros(size=(1, num_heads, out_feats)))
        self.negative_slope=negative_slope
        
        self.reset_parameters()
    
    
    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.W, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        

    def forward(self,row_ptr,col_ind,col_ptr,row_ind,feat,save_memory=True):

        h=torch.matmul(feat,self.W).view(-1,self.num_heads,self.out_feats)

        attn_row = (self.attn_l * h).sum(dim=-1)
        attn_col = (self.attn_r * h).sum(dim=-1)
        
        out=fused_gat_op(attn_row,attn_col,row_ptr,col_ind,col_ptr,row_ind,self.negative_slope,h,save_memory)
            
        return out


class GATConv_stash(nn.Module): # our gat layer
    def __init__(self,
                in_feats,
                out_feats,
                num_heads,
                negative_slope=0.2,

                ):
        super(GATConv_stash,self).__init__()
        self.in_feats=in_feats
        self.out_feats=out_feats
        self.num_heads=num_heads
        self.W = nn.Parameter(torch.FloatTensor(in_feats, out_feats * num_heads))
        self.attn_l = nn.Parameter(torch.zeros(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.zeros(size=(1, num_heads, out_feats)))
        self.negative_slope=negative_slope
        
        self.reset_parameters()
    
    
    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.W, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        

    def forward(self,row_ptr,col_ind,col_ptr,row_ind,feat,permute):

        h=torch.matmul(feat,self.W).view(-1,self.num_heads,self.out_feats)

        attn_row = (self.attn_l * h).sum(dim=-1)
        attn_col = (self.attn_r * h).sum(dim=-1)
        
        out=fused_gat_stash_op(attn_row,attn_col,row_ptr,col_ind,col_ptr,row_ind,self.negative_slope,h,permute)
            
        return out



class GATConv_fusescatter(nn.Module): # our gat layer
    def __init__(self,
                in_feats,
                out_feats,
                num_heads,
                negative_slope=0.2,

                ):
        super(GATConv_fusescatter,self).__init__()
        self.in_feats=in_feats
        self.out_feats=out_feats
        self.num_heads=num_heads
        self.W = nn.Parameter(torch.FloatTensor(in_feats, out_feats * num_heads))
        self.attn_l = nn.Parameter(torch.zeros(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.zeros(size=(1, num_heads, out_feats)))
        self.negative_slope=negative_slope
        
        self.reset_parameters()
    
    
    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.W, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        

    def forward(self,row_ptr,col_ind,col_ptr,row_ind,feat,permute):

        h=torch.matmul(feat,self.W).view(-1,self.num_heads,self.out_feats)

        attn_row = (self.attn_l * h).sum(dim=-1)
        attn_col = (self.attn_r * h).sum(dim=-1)
        
        out=fused_gat_fusescatter(attn_row,attn_col,row_ptr,col_ind,col_ptr,row_ind,self.negative_slope,h,permute)
            
        return out


import dgl.function as fn
from dgl.ops.edge_softmax import edge_softmax
class GATConv_dgl(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 negative_slope=0.2):
        super(GATConv_dgl, self).__init__()
        self._num_heads = num_heads
        self._in_feats= in_feats
        self._out_feats = out_feats

        self.fc = nn.Linear(
                self._in_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)



    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat):
        with graph.local_scope():             
            feat_src = feat_dst = self.fc(feat).view(-1, self._num_heads, self._out_feats)
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            # compute softmax
            graph.edata['a'] = edge_softmax(graph, e)
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            
            return rst

class GATConv_000(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 negative_slope=0.2):
        super(GATConv_000, self).__init__()
        self._num_heads = num_heads
        self._in_feats= in_feats
        self._out_feats = out_feats

        self.fc = nn.Linear(
                self._in_feats, out_feats * num_heads, bias=False)
        self.attn = nn.Parameter(torch.FloatTensor(size=(1, num_heads, 2*out_feats)))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn, gain=gain)


    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat):
        with graph.local_scope():             
            feat= self.fc(feat).view(-1, self._num_heads, self._out_feats)
            temp=torch.zeros(graph.num_nodes(),self._num_heads,self._out_feats,device=feat.device)
            graph.srcdata['ft']=feat
            graph.srcdata['ft0']=torch.cat((feat,temp),dim=2)
            graph.dstdata['ft0']=torch.cat((temp,feat),dim=2)
            graph.apply_edges(fn.u_add_v('ft0','ft0','e0'))
            graph.edata['e']=(self.attn*graph.edata['e0']).sum(dim=-1).unsqueeze(-1)
            e = self.leaky_relu(graph.edata.pop('e'))
            # compute softmax
            graph.edata['a'] = edge_softmax(graph, e)
            
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # print(rst.shape)
            
            return rst


from operators.edge_softmax import csr_edge_softmax
from operators.u_add_v import u_add_v_op
from operators.mhspmm import csrmhspmm
from operators.spmm import csrspmm
import dgl
class GATConv_no_fusion(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 negative_slope=0.2):
        super(GATConv_no_fusion, self).__init__()
        self._num_heads = num_heads
        self._in_feats= in_feats
        self._out_feats = out_feats

        self.fc = nn.Linear(
                self._in_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)



    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, row_ptr,col_ind,col_ptr,row_ind,feat,permute,g):                   
        feat_src = feat_dst = self.fc(feat).view(-1, self._num_heads, self._out_feats)

        el = (feat_src * self.attn_l).sum(dim=-1)
        er = (feat_dst * self.attn_r).sum(dim=-1)
        # print(el.shape)
        # e=u_add_v_op(row_ptr,col_ind,el,er)
        e=dgl.ops.u_add_v(g,er,el)
        # print(torch.abs(e_our-e).sum())
        # num_edge=col_ind.shape[0]
        # head=self._num_heads
        # e=torch.randn(num_edge,head,device=0)
        e = self.leaky_relu(e)
        e=csr_edge_softmax(row_ptr,e)
        rst=csrmhspmm(row_ptr,col_ind,col_ptr,row_ind,permute,feat_src,e,)
        # rst=csrspmm(row_ptr,col_ind,feat_src.view(-1,self._out_feats),e).view(-1,1,self._out_feats)
        
        return rst