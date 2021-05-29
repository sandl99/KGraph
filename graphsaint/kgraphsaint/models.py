from operator import sub
import os
from numpy.lib.arraysetops import isin
from torch._C import device
from torch.functional import Tensor
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

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
        # self.rel = torch.nn.Embedding(self.num_rel + 1, args.dim, padding_idx=0)
        self.args = args
        self.n_iter = args.n_iter

        self.aggregator = nn.ModuleList()
        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                self.aggregator.append(Aggregator(args.batch_size, args.dim, args.aggregator, nn.Tanh()))
            else:
                self.aggregator.append(Aggregator(args.batch_size, args.dim, args.aggregator, nn.ReLU()))
        self.device = device
        self.dim = args.dim

    def forward(self, u, v, reserve_node=None, node=None, subgraph=None):
        """
        input: u, v are batch sized indices for users and items
        u, v: [batch]
        """
        entities = self.ent(node)
        user_embed = self.usr(u)
        for i in range(self.n_iter):
            nei_entities = torch.sparse.mm(subgraph, entities)
            entities = self.aggregator[i](entities, nei_entities)
        if self.training:
            v = [reserve_node[i.item()] for i in v]
            v = torch.LongTensor(v).to(self.device)
        item_embedding = entities[v]
        inner_product = user_embed * item_embedding
        return inner_product.sum(dim=1)
    

