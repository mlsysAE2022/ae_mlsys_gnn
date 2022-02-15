from torch import nn
import sys
sys.path.append('..')
from operators.fused_edgeconv import fused_edgeconv_op

class EdgeConv(nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 batch_norm=False,
                 allow_zero_in_degree=False):
        super(EdgeConv, self).__init__()
        self.batch_norm = batch_norm
        self._allow_zero_in_degree = allow_zero_in_degree

        self.theta = nn.Linear(in_feat, out_feat,bias=False)
        self.phi = nn.Linear(in_feat, out_feat,bias=False)

        if batch_norm:
            self.bn = nn.BatchNorm1d(out_feat)

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, k,src_ind,feat):
        h_theta=self.theta(feat)
        h_phi=self.phi(feat)
        h_src=h_theta
        h_dst=h_phi-h_theta
        result=fused_edgeconv_op(k,src_ind,h_src,h_dst)
        
        return result






import dgl.function as fn
class EdgeConv_dgl(nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 batch_norm=False,
                 allow_zero_in_degree=False):
        super(EdgeConv_dgl, self).__init__()
        self.batch_norm = batch_norm
        self._allow_zero_in_degree = allow_zero_in_degree

        self.theta = nn.Linear(in_feat, out_feat)
        self.phi = nn.Linear(in_feat, out_feat)

        if batch_norm:
            self.bn = nn.BatchNorm1d(out_feat)

    def set_allow_zero_in_degree(self, set_value):

        self._allow_zero_in_degree = set_value

    def forward(self, g, feat):
        with g.local_scope():           
            g.srcdata['x'] = feat
            g.dstdata['x'] = feat
            g.apply_edges(fn.v_sub_u('x', 'x', 'theta'))
            g.edata['theta'] = self.theta(g.edata['theta'])
            g.dstdata['phi'] = self.phi(g.dstdata['x'])            
            g.update_all(fn.e_add_v('theta', 'phi', 'e'), fn.max('e', 'x'))
            return g.dstdata['x']


class EdgeConv_reorg(nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 batch_norm=False,
                 allow_zero_in_degree=False):
        super(EdgeConv_reorg, self).__init__()
        self.batch_norm = batch_norm
        self._allow_zero_in_degree = allow_zero_in_degree

        self.theta = nn.Linear(in_feat, out_feat)
        self.phi = nn.Linear(in_feat, out_feat)

        if batch_norm:
            self.bn = nn.BatchNorm1d(out_feat)

    def set_allow_zero_in_degree(self, set_value):

        self._allow_zero_in_degree = set_value

    def forward(self, g, feat):
        with g.local_scope():
            h_theta=self.theta(feat)
            h_phi=self.phi(feat)        
            g.srcdata['x'] = h_theta
            g.dstdata['x'] = h_phi-h_theta               
            g.update_all(fn.u_add_v('x', 'x', 'e'),fn.max('e', 'x'))
            return g.dstdata['x']