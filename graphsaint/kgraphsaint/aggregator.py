import torch


class Aggregator(torch.nn.Module):
    """
    Aggregator class
    Mode in ['sum', 'concat', 'neighbor']
    """
    def __init__(self, batch_size, dim, aggregator, act):
        super(Aggregator, self).__init__()
        self.batch_size = batch_size
        self.dim = dim
        if aggregator == 'concat':
            self.weights = torch.nn.Linear(2 * dim, dim, bias=True)
        elif aggregator == 'sum':
            self.weights = torch.nn.Linear(dim, dim, bias=True)
        else:
            raise NotImplemented
        self.aggregator = aggregator
        self.act = act

    def forward(self, self_vectors, neighbor_vectors):
        if self.aggregator == 'concat':
            res = torch.cat((self_vectors, neighbor_vectors), dim=1)
        elif self.aggregator == 'sum':
            res = self_vectors + neighbor_vectors
        else:
            res = None
        return self.act(self.weights(res))
