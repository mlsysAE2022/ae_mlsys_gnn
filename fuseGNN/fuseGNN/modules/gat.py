import torch
import torch.nn.functional as F
from fuseGNN.convs import geoGATConv, refGATConv, garGATConv, gasGATConv
from fuseGNN.utils import LrSchedular


class GAT(torch.nn.Module):
    """
    The graph convolutional operator from the `"Graph Attention Networks"
    <https://arxiv.org/pdf/1710.10903.pdf>
    A two-layer GAT model.
    The model is trained for 200 epochs with learning rate 0.01 and early stopped with a window size of 10
    (the validation loss doesn't decrease for 10 consecutive epochs)
    """
    def __init__(self, num_features, hidden, heads, num_classes, drop_rate=0.5, mode='geo', flow='source_to_target'):
        """
        :param num_features: the length of input features
        :param hidden: the length of hidden layer
        :param num_classes: the number of classes
        :param cached: If True, the layer will cache the computation on first execution, and will use the
        cached version for further executions. So it should be only true in transductive learning scenarios
        """
        super(GAT, self).__init__()
        if mode == 'ref':
            self.conv1 = refGATConv(in_channels=num_features, out_channels=int(hidden/heads), heads=heads, dropout=drop_rate)
            self.conv2 = refGATConv(in_channels=hidden, out_channels=num_classes, concat=False, dropout=drop_rate)
        elif mode == 'gar':
            self.conv1 = garGATConv(in_channels=num_features, out_channels=int(hidden/heads), heads=heads, dropout=drop_rate, cached=False, return_mask=False)
            self.conv2 = garGATConv(in_channels=hidden, out_channels=num_classes, concat=False, dropout=drop_rate, cached=False, return_mask=False)
        elif mode == 'gas':
            self.conv1 = gasGATConv(in_channels=num_features, out_channels=int(hidden/heads), heads=heads, dropout=drop_rate, cached=False, return_mask=False)
            self.conv2 = gasGATConv(in_channels=hidden, out_channels=num_classes, concat=False, dropout=drop_rate, cached=False, return_mask=False)
        else:
            self.conv1 = geoGATConv(in_channels=num_features, out_channels=int(hidden/heads), heads=heads, dropout=drop_rate)
            self.conv2 = geoGATConv(in_channels=hidden, out_channels=num_classes, concat=False, dropout=drop_rate)
        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()
        self.mode = mode
        self.drop_rate = drop_rate
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(data.x, p=self.drop_rate, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(input=x, p=self.drop_rate, training=self.training)
        if self.mode == 'gar':
            x = self.conv2(x=x, tar_index_b=self.conv1.cached_tar_index_b, src_index_b=self.conv1.cached_src_index_b, src_ptr=self.conv1.cached_src_ptr)
        elif self.mode == 'gas':
            x = self.conv2(x=x, src_index=self.conv1.cached_src_index, tar_index=self.conv1.cached_tar_index)
        else:
            x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


"""
    Training configurations from 
    "Graph Attention Networks"
    <https://arxiv.org/pdf/1710.10903.pdf> The lines below as cited from the paper:
    > we apply a two-layer GAT model ...
    > The first layer consists of K = 8 attention heads computing F'= 8 features each (for a total of 64 features),
      followed by an exponential linear unit (ELU)
    > The second layer is used for classification: a single attention head that computes C features 
      (where C is the number of classes), followed by a softmax activation.
    > During training, we apply L2 regulariza- tion with λ = 0.0005.
    > Furthermore, dropout with p = 0.6 is applied to both layers’ inputs,
    > normalized attention coefficients
    
    > For PubMed: we have applied K = 8 output attention heads (instead of one), and strengthened the L2 regularization to λ = 0.001. 
      Otherwise, the architecture matches the one used for Cora and Citeseer.
    
    > Both models are initialized using Glorot initialization and trained to minimize cross-entropy on the training nodes 
      using the Adam SGD optimizer with an initial learning rate of 0.01 for Pubmed, and 0.005 for all other datasets. 
    > 
    > We train all models for a maximum of 200 epochs (training iterations) 
      using Adam with a learning rate of 0.01
    > with a patience of 100 epochs
"""
gat_config = {
    'CiteSeer': {
        'drop_rate': 0.6,
        'weight_decay': 5e-4,
        'hidden': 64,
        'lr': 0.005,
        'head': 1,
        'lr_schedular': LrSchedular(init_lr=0.005, mode='constant'),
        'fold': 1,
    },
    'Cora': {
        'drop_rate': 0.6,
        'weight_decay': 5e-4,
        'hidden': 64,
        'lr': 0.005,
        'head': 1,
        'lr_schedular': LrSchedular(init_lr=0.005, mode='constant'),
        'fold': 1,
    },
    'PubMed': {
        'drop_rate': 0.6,
        'weight_decay': 1e-3,
        'hidden': 64,
        'lr': 0.01,
        'head': 1,
        'lr_schedular': LrSchedular(init_lr=0.01, mode='constant'),
        'fold': 1,
    },
    'Reddit': {
        'drop_rate': 0.5,
        'weight_decay': 1e-3,
        'hidden': 128,
        'lr': 0.001,
        'head': 1,
        'lr_schedular': LrSchedular(init_lr=0.01, mode='constant'),
        'fold': 1,
    },
}
