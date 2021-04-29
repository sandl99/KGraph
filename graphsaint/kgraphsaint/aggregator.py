import torch


def softmax(inp, masks, dim):
    # masks = masks * -100000.0
    # inp = inp + masks
    inp_exp = torch.exp(inp)
    inp_exp = inp_exp * (inp != 0).float()
    inp_sum = torch.sum(inp_exp, dim=dim, keepdim=True)
    inp_sum = torch.where(inp_sum != 0, inp_sum, torch.tensor(1, dtype=torch.float, requires_grad=False, device='cuda'))
    inp_softmax = inp_exp / inp_sum
    return inp_softmax    # convert NaN to Zero value


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

    def forward(self, self_vectors, neighbor_vectors, neighbor_relations, masks, user_embeddings, act):
        batch_size = user_embeddings.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, masks, user_embeddings)

        if self.aggregator == 'sum':
            output = (self_vectors + neighbors_agg).view((-1, self.dim))

        elif self.aggregator == 'concat':
            output = torch.cat((self_vectors, neighbors_agg), dim=-1)
            output = output.view((-1, 2 * self.dim))

        else:
            output = neighbors_agg.view((-1, self.dim))

        output = self.weights(output)
        return act(output.view((self.batch_size, -1, self.dim)))

    def _mix_neighbor_vectors(self, neighbor_vectors, neighbor_relations, masks, user_embeddings):
        '''
        This aims to aggregate neighbor vectors
        '''
        # [batch_size, 1, dim] -> [batch_size, 1, 1, dim]
        user_embeddings = user_embeddings.view((self.batch_size, 1, 1, self.dim))

        # [batch_size, -1, n_neighbor, dim] -> [batch_size, -1, n_neighbor]
        user_relation_scores = (user_embeddings * neighbor_relations).sum(dim=-1)
        user_relation_scores_normalized = softmax(user_relation_scores, masks, dim=-1)

        # [batch_size, -1, n_neighbor] -> [batch_size, -1, n_neighbor, 1]
        user_relation_scores_normalized = user_relation_scores_normalized.unsqueeze(dim=-1)

        # [batch_size, -1, n_neighbor, 1] * [batch_size, -1, n_neighbor, dim] -> [batch_size, -1, dim]
        neighbors_aggregated = (user_relation_scores_normalized * neighbor_vectors).sum(dim=2)

        return neighbors_aggregated
