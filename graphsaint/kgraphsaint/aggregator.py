from operator import ne
import torch
from torch._C import layout
from torch_sparse import mul
from torch_scatter import gather_csr, segment_csr
from torch_sparse import SparseTensor

def softmax(src: SparseTensor, dim=1, unsqueeze=True):
    value = src.storage.value()
    rowptr = src.storage.rowptr()

    value_exp = torch.exp(value)
    sum_value_exp = segment_csr(value_exp, rowptr)
    sum_value_exp = gather_csr(sum_value_exp, rowptr)
    res = torch.div(value_exp, sum_value_exp)
    if unsqueeze:
        res = res.unsqueeze(dim=-1)
    return src.set_value(res, layout='csr')

class Aggregator(torch.nn.Module):
    """
    Aggregator class
    Mode in ['sum', 'concat', 'neighbor']
    """
    def __init__(self, batch_size, dim, aggregator):
        super(Aggregator, self).__init__()
        self.batch_size = batch_size
        self.dim = dim
        if aggregator == 'concat':
            self.weights = torch.nn.Linear(2 * dim, dim, bias=True)
        else:
            self.weights = torch.nn.Linear(dim, dim, bias=True)
        self.aggregator = aggregator

    def forward(self, self_vectors, neighbor_vectors, neighbor_relations, neighbor_norms, user_embeddings, act):
        batch_size = user_embeddings.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, neighbor_norms, user_embeddings, self_vectors)

        if isinstance(self_vectors, SparseTensor):
            # dcm = neighbors_agg
            # neighbors_agg = SparseTensor.from_dense(neighbors_agg.view(self.batch_size, -1, self.dim), True)
            pass
        else:
            neighbors_agg = neighbors_agg.unsqueeze(dim=1)

        if self.aggregator == 'sum':
            if isinstance(self_vectors, SparseTensor):
                output = self_vectors.add_nnz(neighbors_agg, layout='csr')
            else:
                output = (self_vectors + neighbors_agg).view((-1, self.dim))
        elif self.aggregator == 'concat':
            raise NotImplementedError
            # output = torch.cat((self_vectors, neighbors_agg), dim=-1)
            # output = output.view((-1, 2 * self.dim)
        else:
            raise NotImplementedError
            output = neighbors_agg.view((-1, self.dim))

        if isinstance(self_vectors, SparseTensor):
            # output = output.to_dense().view((self.batch_size, -1, self.dim))
            # res = SparseTensor.from_dense(output, has_value=True)
            # assert self_vectors.storage.value().shape[0] == res.storage.value().shape[0]
            # output = res
            value = output.storage.value()
            value = act(self.weights(value))
            return output.set_value(value, layout='csr')
        else:
            output = self.weights(output)
            return act(output.view((self.batch_size, -1, self.dim)))


    def _mix_neighbor_vectors(self, neighbor_vectors, neighbor_relations, neighbor_norms, user_embeddings, self_vectors):
        """ old implement
        '''
        This aims to aggregate neighbor vectors
        '''
        # [batch_size, 1, dim] -> [batch_size, 1, 1, dim]
        user_embeddings = user_embeddings.view((self.batch_size, 1, 1, self.dim))

        # [batch_size, -1, n_neighbor, dim] -> [batch_size, -1, n_neighbor]
        user_relation_scores = (user_embeddings * neighbor_relations).sum(dim=-1)
        user_relation_scores_normalized = softmax(user_relation_scores, dim=-1)

        # [batch_size, -1, n_neighbor] -> [batch_size, -1, n_neighbor, 1]
        user_relation_scores_normalized = user_relation_scores_normalized.unsqueeze(dim=-1)

        # [batch_size, -1, n_neighbor, 1] * [batch_size, -1, n_neighbor, dim] -> [batch_size, -1, dim]
        neighbors_aggregated = (user_relation_scores_normalized * neighbor_vectors).sum(dim=2)

        return neighbors_aggregated
        """
        # user_embeddings: [batch_size, 1, dim], user_relation_scores: [batch_size, neighbor, dim]
        assert neighbor_relations.size(0) % self.batch_size == 0
        user_embeddings = user_embeddings.repeat(neighbor_relations.size(0) // self.batch_size, 1)
        user_embeddings = user_embeddings.view((user_embeddings.shape[0], 1, self.dim))
        user_relation_scores = neighbor_relations.mul(user_embeddings)

        # user_relation_scores: [batch_size, neighbor, 1]
        user_relation_scores = user_relation_scores.set_value(user_relation_scores.sum(dim=-1), layout='csr')     
        # softmax to normalize [batch_size, n_neighbor, 1]
        user_relation_scores_normalized = softmax(user_relation_scores, unsqueeze=True)
        # apply relation normalized
        rel_val = user_relation_scores_normalized.storage.value()
        nei_val = neighbor_vectors.storage.value()
        # neighbor_vectors = neighbor_vectors.set_value(nei_val * rel_val, layout='csr')
        nei_val = nei_val * rel_val
        # apply norm aggregate parameter
        if neighbor_norms is not None:
            norm_aggr_parameter = neighbor_norms.storage.value().unsqueeze(1)
            neighbor_vectors = neighbor_vectors.set_value(nei_val * norm_aggr_parameter, layout='csr')
        else:
            neighbor_vectors = neighbor_vectors.set_value(nei_val, layout='csr')
        # user_relation_normalized
        if isinstance(self_vectors, SparseTensor):
            neighbors_aggregated = segment_csr(neighbor_vectors.storage.value(), torch.unique(neighbor_vectors.storage.rowptr()), reduce='mean')
        else:
            neighbors_aggregated = neighbor_vectors.mean(dim=1)
        # assert nei_val.shape[0] == SparseTensor.from_dense(neighbors_aggregated.view(256, -1, 32), has_value=True).storage.value().shape[0]
        return neighbors_aggregated
