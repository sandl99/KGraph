import os
from numpy.lib.arraysetops import isin
from torch._C import device
from torch.functional import Tensor
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
from torch import nn
import torch.nn.functional as F
from graphsaint.kgraphsaint.aggregator import Aggregator
from graphsaint.kgraphsaint.utils import index_select
from torch_sparse import SparseTensor, tensor


class KGraphSAINT(nn.Module):
    def __init__(self, num_usr, num_ent, num_rel, args, device='cuda'):
        super(KGraphSAINT, self).__init__()
        self.num_usr = num_usr
        self.num_ent = num_ent
        self.num_rel = num_rel

        self.usr = torch.nn.Embedding(self.num_usr, args.dim)
        self.ent = torch.nn.Embedding(self.num_ent, args.dim, padding_idx=0)
        self.rel = torch.nn.Embedding(self.num_rel + 1, args.dim, padding_idx=0)
        self.args = args
        self.n_iter = args.n_iter

        self.aggregator = nn.ModuleList()
        for i in range(self.n_iter):
            self.aggregator.append(Aggregator(args.batch_size, args.dim, args.aggregator))
        self.device = device
        # self.n_neighbor = args.neighbor_sample_size
        self.dim = args.dim
        self.norm_aggr = None
    
    def set_norm_aggr(self, norm_aggr):
        self.norm_aggr = norm_aggr

    def forward(self, u, v, reserve_node=None, node=None, adj=None, rel=None, edge_idx=None, train_mode=True):
        """
        input: u, v are batch sized indices for users and items
        u, v: [batch]
        """
        # convert u, v to [batch, 1] shape
        u = u.view((-1, 1))
        #
        if train_mode:
            v = [reserve_node[i.item()] for i in v]
            v = torch.LongTensor(v).to(self.device)
            assert edge_idx != None
        v = v.view((-1, 1))
        if node is not None:
            node = node.type(torch.long)
        # [batch_size, dim]
        batch_size = v.shape[0]
        user_embeddings = self.usr(u).squeeze(dim=1)
        entities, relations, aggregation_norms = self._get_neighbors(v, node, adj, rel, edge_idx, train_mode)
        item_embeddings = self._aggregate(user_embeddings, entities, relations, aggregation_norms, batch_size)
        scores = (user_embeddings * item_embeddings).sum(dim=1)
        return scores

    def _get_neighbors(self, v, node, adj, rel, edge_idx, train_mode):
        '''
        v is batch sized indices for items
        v: [batch_size, 1]
        '''
        batch_size = v.shape[0]
        entities = [v]
        relations = []
        aggregation_norms = []
        # node = torch.LongTensor(node).to(self.device)
        for h in range(self.n_iter):
            neighbor_entities = index_select(adj, entities[h])
            # neighbor_relations = index_select(rel, entities[h])
            entities.append(neighbor_entities)
            # relations.append(neighbor_relations)
            if train_mode:
                neighbor_aggregations = index_select(edge_idx, entities[h])
                aggregation_norms.append(neighbor_aggregations)
            # masks.append(torch.where(neighbor_relations == 0, torch.tensor(1., dtype=torch.float, device=self.device), torch.tensor(0., dtype=torch.float, device=self.device)))
        if train_mode:
            n_entities = [None] * len(entities)
            for i, h in enumerate(entities):
                if isinstance(h, SparseTensor):
                    val = h.storage.value()
                    val = node[val.type(torch.long)].to(self.device)
                    _h = h.set_value(val, layout='csr')
                    n_entities[i] = _h
                elif isinstance(h, Tensor):
                    n_entities[i] = node[h].to(self.device)
                else:
                    raise ValueError
            entities = n_entities
        return entities, relations, aggregation_norms

    def _aggregate(self, user_embeddings, entities, relations, aggregation_norms, batch_size):
        '''
        Make item embeddings by aggregating neighbor vectors
        '''
        train_mode = (len(aggregation_norms) != 0)
        entity_vectors = [None] * len(entities)
        relation_vectors = [None] * (len(entities) -1)
        aggr_norm_vectors = [None] * (len(entities) - 1)

        for i, entity in enumerate(entities):
            if isinstance(entity, SparseTensor):
                val = entity.storage.value()
                assert (val == 0).sum() == 0
                val = self.ent(val)
                entity_vectors[i] = entity.set_value(val, layout='csr')
            else:
                entity_vectors[i] = self.ent(entity)
        # for i, relation in enumerate(relations):
        #     if isinstance(relation, SparseTensor):
        #         val = relation.storage.value()
        #         assert (val == 0).sum() == 0
        #         val = self.rel(val)
        #         relation_vectors[i] = relation.set_value(val, layout='csr')
        #     else:
        #         relation_vectors[i] = self.rel(relation)
        if train_mode:
            for i, aggr in enumerate(aggregation_norms):
                aggr_norm_vectors[i] = aggregation_norms[i]

        n_neighbor = entities[1].size(1)

        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                act = torch.tanh
            else:
                act = torch.relu

            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                vector = self.aggregator[i](
                    self_vectors=entity_vectors[hop],
                    neighbor_vectors=entity_vectors[hop + 1],
                    neighbor_relations=relation_vectors[hop],
                    neighbor_norms=aggr_norm_vectors[hop],
                    user_embeddings=user_embeddings,
                    act=act)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        return entity_vectors[0].view((batch_size, self.dim))
