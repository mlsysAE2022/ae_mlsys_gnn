import torch
import torch.nn.functional as F
from fuseGNN.convs import geoGCNConv, refGCNConv, garGCNConv, gasGCNConv
from fuseGNN.utils import LrSchedular


modules = {
    'geo': geoGCNConv,
    'ref': refGCNConv,
    'gar': garGCNConv,
    'gas': gasGCNConv
}


class GCN(torch.nn.Module):
    """
    The graph convolutional operator from the `"Semi-supervised Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>
    A two-layer GCN model.
    The model is trained for 200 epochs with learning rate 0.01 and early stopped with a window size of 10
    (the validation loss doesn't decrease for 10 consecutive epochs)
    """
    def __init__(self, num_features, hidden, num_classes, cached=True, drop_rate=0.5, mode='geo', flow='source_to_target'):
        """
        :param num_features: the length of input features
        :param hidden: the length of hidden layer
        :param num_classes: the number of classes
        :param cached: If True, the layer will cache the computation on first execution, and will use the
        cached version for further executions. So it should be only true in transductive learning scenarios
        """
        super(GCN, self).__init__()
        self.GCNConv = modules[mode]
        self.mode = mode
        self.conv1 = self.GCNConv(in_channels=num_features, out_channels=hidden, cached=cached, flow=flow)
        self.conv2 = self.GCNConv(in_channels=hidden, out_channels=num_classes, cached=cached, flow=flow)
        
        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()
        self.drop_rate = drop_rate
    
    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(input=x, p=self.drop_rate, training=self.training)
        if self.mode == 'gar':
            x = self.conv2(x=x, edge_weight=self.conv1.cached_edge_weight_f, self_edge_weight=self.conv1.cached_self_edge_weight,
                           tar_ptr=self.conv1.cached_tar_ptr, src_index=self.conv1.cached_src_index,
                           src_ptr=self.conv1.cached_src_ptr, tar_index=self.conv1.cached_tar_index,
                           edge_weight_b=self.conv1.cached_edge_weight_b)
        elif self.mode == 'gas':
            x = self.conv2(x=x, edge_weight=self.conv1.cached_edge_weight, src_index=self.conv1.cached_src_index,
                           tar_index=self.conv1.cached_tar_index, self_edge_weight=self.conv1.cached_self_edge_weight)
        else:
            x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


"""
    Training configurations from 
    "Semi-supervised Classification with Graph Convolutional Networks"
    <https://arxiv.org/pdf/1609.02907.pdf> The lines below as cited from the paper:
    > we train a two-layer GCN as ...
    > For the citation network datasets, we optimize hyperparameters on Cora only 
      and use the same set of parameters for Citeseer and Pubmed.
    > We train all models for a maximum of 200 epochs (training iterations) 
      using Adam with a learning rate of 0.01
    > We used the following sets of hyperparameters for Citeseer, Cora and Pubmed: 
      0.5 (dropout rate), 5 · 10−4 (L2 regularization) and 16 (number of hid- den units); 
      and for NELL: 0.1 (dropout rate), 1 · 10−5 (L2 regularization) and 64 (number of hidden units).
"""
gcn_config = {
    'CiteSeer': {
        'drop_rate': 0.5,
        'weight_decay': 4e-4,
        'hidden': 16,
        'lr': 0.01,
        'lr_schedular': LrSchedular(init_lr=0.01, mode='constant'),
        'fold': 1,
    },
    'Cora': {
        'drop_rate': 0.5,
        'weight_decay': 4e-4,
        'hidden': 16,
        'lr': 0.01,
        'lr_schedular': LrSchedular(init_lr=0.01, mode='constant'),
        'fold': 1,
    },
    'PubMed': {
        'drop_rate': 0.5,
        'weight_decay': 4e-4,
        'hidden': 16,
        'lr': 0.01,
        'lr_schedular': LrSchedular(init_lr=0.01, mode='constant'),
        'fold': 1,
    },
    'Nell': {
        'drop_rate': 0.1,
        'weight_decay': 1e-5,
        'hidden': 64,
        'lr': 0.01,
        'lr_schedular': LrSchedular(init_lr=0.01, mode='constant'),
        'fold': 1,
    },
    'Reddit': {
        'drop_rate': 0.5,
        'weight_decay': 1e-5,
        'hidden': 128,
        'lr': 0.01,
        'lr_schedular': LrSchedular(init_lr=0.01, mode='constant'),
        'fold': 1,
    }
}
