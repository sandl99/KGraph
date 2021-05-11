import torch
from torch import nn
import torch.nn.functional as F
from graphsaint.kgraphsaint.aggregator import Aggregator


class KGraphSAINT(nn.Module):
    def __init__(self, num_usr, num_ent, num_rel, args, device='cuda'):
        super(KGraphSAINT, self).__init__()
        self.num_usr = num_usr
        self.num_ent = num_ent
        self.num_rel = num_rel

        self.usr = torch.nn.Embedding(self.num_usr, args.dim)
        self.ent = torch.nn.Embedding(self.num_ent, args.dim)
        self.rel = torch.nn.Embedding(self.num_rel + 1, args.dim)
        self.args = args
        self.n_iter = args.n_iter

        self.aggregator = nn.ModuleList()
        for i in range(self.n_iter):
            self.aggregator.append(Aggregator(args.batch_size, args.dim, args.aggregator))
        self.device = device
        # self.n_neighbor = args.neighbor_sample_size
        self.dim = args.dim

    def forward(self, u, v, reserve_node=None, node=None, adj=None, rel=None, train_mode=True):
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
        v = v.view((-1, 1))

        # [batch_size, dim]
        batch_size = v.shape[0]
        user_embeddings = self.usr(u).squeeze(dim=1)
        entities, relations = self._get_neighbors(v, node, adj, rel, train_mode)
        item_embeddings = self._aggregate(user_embeddings, entities, relations, batch_size)
        scores = (user_embeddings * item_embeddings).sum(dim=1)
        return torch.sigmoid(scores)

    def _get_neighbors(self, v, node, adj, rel, train_mode):
        '''
        v is batch sized indices for items
        v: [batch_size, 1]
        '''
        batch_size = v.shape[0]
        entities = [v]
        relations = []
        # node = torch.LongTensor(node).to(self.device)
        for h in range(self.n_iter):
            neighbor_entities = torch.LongTensor(adj[entities[h]]).view((batch_size, -1)).to(self.device)
            neighbor_relations = torch.LongTensor(rel[entities[h]]).view((batch_size, -1)).to(self.device)
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        if train_mode:
            entities = [node[h].to(self.device) for h in entities]
        return entities, relations

    def _aggregate(self, user_embeddings, entities, relations, batch_size):
        '''
        Make item embeddings by aggregating neighbor vectors
        '''
        entity_vectors = [self.ent(entity) for entity in entities]
        relation_vectors = [self.rel(relation) for relation in relations]

        n_neighbor = entities[1].shape[1]

        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                act = torch.tanh
            else:
                act = torch.sigmoid

            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                vector = self.aggregator[i](
                    self_vectors=entity_vectors[hop],
                    neighbor_vectors=entity_vectors[hop + 1].view((batch_size, -1, n_neighbor, self.dim)),
                    neighbor_relations=relation_vectors[hop].view((batch_size, -1, n_neighbor, self.dim)),
                    user_embeddings=user_embeddings,
                    act=act)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        return entity_vectors[0].view((batch_size, self.dim))
