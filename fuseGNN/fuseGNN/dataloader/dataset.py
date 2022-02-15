import torch
import torch_geometric.transforms as T
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import DataLoader, DenseDataLoader
from fuseGNN.dataloader import Citations, graph_kernel_dataset


def k_fold(dataset, folds):
    """
    Separate the dataset into k folds
    """
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        # split: generate indices to split data
        # _: train indices, idx: test indices
        test_indices.append(torch.from_numpy(idx))

    val_indices = [test_indices[i - 1] for i in range(folds)]
    # the validation set is just a permutated list of test set

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero().view(-1))

    return train_indices, test_indices, val_indices  # just returns three lists, each one has length fold.


class DataProvider:
    def __init__(self, data, data_path, task, **kwargs):
        path = data_path + data

        if data in ['CiteSeer', 'Cora', 'PubMed']:
            """ Citation Networks: CiteSeer, Cora, and PubMed
            Description:
                these datasets contain sparse bag-of-words feature vectors for each document and a list of citation 
                links between documents. The citation links are treated as undirected edges
            """
            assert task == "node_class", "%s is a dataset for node classification" % data
            self.dataset = Citations(path, data, T.NormalizeFeatures())
            self.data = self.dataset[0]
            # these citation networks are small enough fit into the GPUs, but they still have enough nodes for
            # training, validation, and test.

        elif data in ['COLLAB', 'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'REDDIT-MULTI5K', 'PROTEINS', 'MUTAG',
                      'PTC', 'NCI1']:
            """ Benchmark Data Sets for Graph Kernels
            Description:
                these datasets are the benchmarks for graph classification, the detailed information is available 
                at <https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets>
            """
            assert task == "graph_class", "%s is a dataset for graph classification" % data
            self.dataset = graph_kernel_dataset(data, path)
            self.train_ids, self.test_ids, self.val_ids = k_fold(self.dataset, kwargs['fold'])
            self.batch_size = kwargs['batch_size']
            # these datasets are much smaller. For instance, MUTAG only has 188 different graphs. So unlike the above
            # node classification, here we need cross validation.
        else:
            raise NameError('unknown dataset')

    def get_cross_validation_loader(self, fold):
        (train_idx, test_idx, val_idx) = self.train_ids[fold], self.test_ids[fold], self.val_ids[fold]
        train_dataset = self.dataset[train_idx]
        test_dataset = self.dataset[test_idx]
        val_dataset = self.dataset[val_idx]
        if 'adj' in train_dataset[0]:
            train_loader = DenseDataLoader(train_dataset, self.batch_size, shuffle=True)
            val_loader = DenseDataLoader(val_dataset, self.batch_size, shuffle=False)
            test_loader = DenseDataLoader(test_dataset, self.batch_size, shuffle=False)
        else:
            train_loader = DataLoader(train_dataset, self.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, self.batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, self.batch_size, shuffle=False)
        return train_loader, val_loader, test_loader
