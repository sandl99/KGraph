import torch
import scipy as sp
import numpy as np
import argparse
from graphsaint.kgraphsaint import loader
from graphsaint.graph_samplers import *
import time
from graphsaint.norm_aggr import *
from graphsaint.utils import *
import math


def _adj_to_sparse_matrix(adj_ent, adj_rel, n_ent, type):
    row = adj_ent[0]
    col = adj_ent[1]
    value = adj_rel

    remove_dup = []
    tuple_dup = set()
    for cnt, (i, j) in enumerate(zip(row, col)):
        if (i, j) in tuple_dup:
            remove_dup.append(cnt)
        else:
            tuple_dup.add((i, j))
    value = np.delete(value, remove_dup)
    row = np.delete(row, remove_dup)
    col = np.delete(col, remove_dup)

    if type == 'torch':
        indices = np.vstack((row, col))
        i = torch.LongTensor(indices)
        v = torch.LongTensor(value)
        return torch.sparse.FloatTensor(i, v, torch.Size((n_ent, n_ent)))
    elif type == 'csr':
        return sp.csr_matrix((value, (row, col)), shape=(n_ent, n_ent))


def build_adj_matrix(node, csr, neighbor_size=50):
    assert node[0] != 0
    neighbor = np.zeros(shape=len(node) + 1, dtype=np.int64)
    for i in range(1, neighbor.shape[0]):
        neighbor[i] = len(csr.getrow(i - 1).indices)
    n_adj = np.zeros(shape=(neighbor.shape[0], neighbor_size), dtype=np.int64)
    for i in range(1, neighbor.shape[0]):
        if neighbor[i] < neighbor_size:
            n_adj[i] = np.append(csr.getrow(i - 1).indices, np.zeros(shape=neighbor_size - neighbor[i]))
        else:
            n_adj[i] = np.random.choice(csr.getrow(i - 1).indices, n_adj.shape[1], replace=False)
    return n_adj


def build_rel_matrix(node, csr, adj):
    n_rel = np.zeros(shape=adj.shape, dtype=np.int64)
    for i in range(n_rel.shape[0]):
        for j in range(n_rel.shape[1]):
            if adj[i, j] == 0:
                n_rel[i, j] = 0
            else:
                n_rel[i, j] = csr[i - 1, adj[i, j]]
    return n_rel


def statistic(inptrs):
    l1 = []
    for inptr in inptrs:
        tmp = [inptr[i + 1] - inptr[i] for i in range(len(inptr) - 1)]
        l1.append(np.array(tmp))
    l1 = np.array(l1)
    i = 0


class Minibatch:
    """
    Provides minibatches for the trainer or evaluator. This class is responsible for
    calling the proper graph sampler and estimating normalization coefficients.
    """

    def __init__(self, adj_entity, adj_relation, n_entity, n_relation, args, is_cuda=False):
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.adj_full = _adj_to_sparse_matrix(adj_entity, adj_relation, n_entity, 'csr')
        # self.is_cuda = is_cuda
        #
        # if self.is_cuda:
        #     print(torch.cuda.is_available())
        #     self.adj_full = self.adj_full.to('cuda')

        self.node_subgraph = None
        self.batch_num = -1
        self.total_node = n_entity

        self.method_sample = None
        self.subgraphs_remaining_indptr = []
        self.subgraphs_remaining_indices = []
        self.subgraphs_remaining_data = []
        self.subgraphs_remaining_nodes = []
        self.subgraphs_remaining_edge_index = []

        self.norm_loss_test = np.zeros(self.adj_full.shape[0])
        _denom = n_entity
        self.norm_loss_test[np.arange(n_entity)] = 1. / _denom
        self.norm_loss_test = torch.from_numpy(self.norm_loss_test)

        tmp1, tmp2 = self.adj_full.nonzero()
        value, count = np.unique(tmp1, return_counts=True)
        self.deg_train = count
        self.args = args
        # if self.is_cuda:
        #     self.norm_loss_test = self.norm_loss_test.to('cuda')

    def set_sampler(self, train_phases):
        """
        Pick the proper graph sampler. Run the warm-up phase to estimate
        loss / aggregation normalization coefficients.

        Inputs:
            train_phases       dict, config / params for the graph sampler

        Outputs:
            None
        """
        self.subgraphs_remaining_indptr = []
        self.subgraphs_remaining_indices = []
        self.subgraphs_remaining_data = []
        self.subgraphs_remaining_nodes = []
        self.subgraphs_remaining_edge_index = []
        self.method_sample = train_phases['sampler']

        if self.method_sample == 'mrw':
            raise NotImplementedError
        elif self.method_sample == 'rw':
            self.size_subg_budget = 3000 * 2
            self.graph_sampler = rw_sampling(
                self.adj_full,
                np.arange(1, self.n_entity),
                self.size_subg_budget,
                3000,
                2,
            )
        elif self.method_sample == 'node':
            self.size_subg_budget = train_phases['size_subg_edge']
            self.graph_sampler = edge_sampling(
                self.adj_full,
                np.arange(1, self.n_entity),
                train_phases['size_subg_edge']
            )
        elif self.method_sample == 'edge':
            self.size_subg_budget = train_phases['size_subg_edge'] * 2
            self.graph_sampler = edge_sampling(
                self.adj_full,
                np.arange(1, self.n_entity),
                train_phases['size_subg_edge']
            )
        else:
            raise NotImplementedError

        self.norm_loss_train = np.zeros(self.adj_full.shape[0])
        self.norm_aggr_train = np.zeros(self.adj_full.size).astype(np.float32)

        # -------------------------------------------------------------
        # BELOW: estimation of loss / aggregation normalization factors
        # -------------------------------------------------------------
        # For some special sampler, no need to estimate norm factors, we can calculate
        # the node / edge probabilities directly.
        # However, for integrity of the framework, we follow the same procedure
        # for all samplers:
        #   1. sample enough number of subgraphs
        #   2. update the counter for each node / edge in the training graph
        #   3. estimate norm factor alpha and lambda
        tot_sampled_nodes = 0
        # while True:
        #     self.par_graph_sample('train')
        #     tot_sampled_nodes = sum([len(n) for n in self.subgraphs_remaining_nodes])
        #     if tot_sampled_nodes > self.sample_coverage * self.node_train.size:
        #         break
        # print()
        # num_subg = len(self.subgraphs_remaining_nodes)
        # for i in range(num_subg):
        #     self.norm_aggr_train[self.subgraphs_remaining_edge_index[i]] += 1
        #     self.norm_loss_train[self.subgraphs_remaining_nodes[i]] += 1
        # assert self.norm_loss_train[self.node_val].sum() + self.norm_loss_train[self.node_test].sum() == 0
        # for v in range(self.adj_train.shape[0]):
        #     i_s = self.adj_train.indptr[v]
        #     i_e = self.adj_train.indptr[v + 1]
        #     val = np.clip(self.norm_loss_train[v] / self.norm_aggr_train[i_s : i_e], 0, 1e4)
        #     val[np.isnan(val)] = 0.1
        #     self.norm_aggr_train[i_s : i_e] = val
        # self.norm_loss_train[np.where(self.norm_loss_train==0)[0]] = 0.1
        # self.norm_loss_train[self.node_val] = 0
        # self.norm_loss_train[self.node_test] = 0
        # self.norm_loss_train[self.node_train] = num_subg / self.norm_loss_train[self.node_train] / self.node_train.size
        # self.norm_loss_train = torch.from_numpy(self.norm_loss_train.astype(np.float32))
        # if self.use_cuda:
        #     self.norm_loss_train = self.norm_loss_train.cuda()

    def par_graph_sample(self, phase):
        """
        Perform graph sampling in parallel. A wrapper function for graph_samplers.py
        """
        t0 = time.time()
        _indptr, _indices, _data, _v, _edge_index = self.graph_sampler.par_sample(phase)
        t1 = time.time()
        print('sampling 200 subgraphs:   time = {:.3f} sec'.format(t1 - t0), end="\r")
        self.subgraphs_remaining_indptr.extend(_indptr)
        self.subgraphs_remaining_indices.extend(_indices)
        self.subgraphs_remaining_data.extend(_data)
        self.subgraphs_remaining_nodes.extend(_v)
        self.subgraphs_remaining_edge_index.extend(_edge_index)
        # statistic(_indptr)

    def one_batch(self, mode='train'):
        """
        Generate one minibatch for trainer. In the 'train' mode, one minibatch corresponds
        to one subgraph of the training graph. In the 'val' or 'test' mode, one batch
        corresponds to the full graph (i.e., full-batch rather than minibatch evaluation
        for validation / test sets).

        Inputs:
            mode                str, can be 'train', 'val', 'test' or 'valtest'

        Outputs:
            node_subgraph       np array, IDs of the subgraph / full graph nodes
            adj                 scipy CSR, adj matrix of the subgraph / full graph
            norm_loss           np array, loss normalization coefficients. In 'val' or
                                'test' modes, we don't need to normalize, and so the values
                                in this array are all 1.
        """
        if mode in ['val', 'test', 'valtest']:
            self.node_subgraph = np.arange(self.adj_full_norm.shape[0])
            adj = self.adj_full_norm
        else:
            assert mode == 'train'
            if len(self.subgraphs_remaining_nodes) == 0:
                self.par_graph_sample('train')
                print()

            self.node_subgraph = self.subgraphs_remaining_nodes.pop()
            self.size_subgraph = len(self.node_subgraph)
            self.subgraphs_remaining_data.pop()
            col = self.subgraphs_remaining_indices.pop()
            indptr = self.subgraphs_remaining_indptr.pop()
            row = []
            for i in range(len(indptr) - 1):
                row.extend([i] * len(col[indptr[i]:indptr[i+1]]))
            t_row = [self.node_subgraph[i] for i in row]
            t_col = [self.node_subgraph[i] for i in col]
            data = [self.adj_full[i, j] for i, j in zip(t_row, t_col)]
            adj = sp.csr_matrix((data, (row, col)), shape=(self.size_subgraph, self.size_subgraph))

            adj_edge_index = self.subgraphs_remaining_edge_index.pop()
            print("{} nodes, {} edges, {} degree".format(self.node_subgraph.size, adj.size,
                                                         adj.size / self.node_subgraph.size))
            # norm_aggr(adj.data, adj_edge_index, self.norm_aggr_train, num_proc=2)
            # adj.data[:] = self.norm_aggr_train[adj_edge_index][:]      # this line is interchangable with the above line
            # adj = adj_norm(adj, deg=self.deg_train[self.node_subgraph])
            # tmp = adj.tocoo()
            # adj = _adj_to_sparse_matrix((tmp.row, tmp.col), tmp.data, self.node_subgraph.size, type='torch')
            # if self.use_cuda:
            # adj = adj.cuda()
            self.batch_num += 1
        # norm_loss = self.norm_loss_test if mode in ['val', 'test', 'valtest'] else self.norm_loss_train
        # norm_loss = norm_loss[self.node_subgraph]
        # return self.node_subgraph, adj, norm_loss
        # t1 = time.time()
        adj_matrix = build_adj_matrix(self.node_subgraph, adj, neighbor_size=self.args.neighbor_sample_size_train)
        # t2 = time.time()
        # print(f'san dcm {t2-t1}')
        rel_matrix = build_rel_matrix(self.node_subgraph, adj, adj_matrix)
        self.node_subgraph = np.insert(self.node_subgraph, 0, 0)
        # print(f'san dcm {time.time() - t2}')
        return self.node_subgraph, adj_matrix, rel_matrix

    def num_training_batches(self):
        return math.ceil(self.total_node / float(self.size_subg_budget))

    # def shuffle(self):
    #     self.total_node = np.random.permutation(self.total_node)
    #     self.batch_num = -1

    def end(self):
        return (self.batch_num + 1) * self.size_subg_budget >= self.total_node


# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', default='movie')
# parser.add_argument('--ratio', default=1)
# parser.add_argument('--neighbor_sample_size', default=8)
# args = parser.parse_args()
# n_entity, n_relation, adj_entity, adj_relation = loader.load_kg(args)
# n_item, train_data = loader.load_rating(args)
# mini = Minibatch(adj_entity, adj_relation, n_entity, n_relation)
# train_phases = {'size_subg_edge': 4000, 'sampler': 'edge'}
# mini.set_sampler(train_phases)
# t0 = time.time()
# n_s, adj = mini.one_batch('train')
# t1 = time.time()
# print(f'san debug {t1-t0}')
# neighbor = np.zeros(shape=len(n_s))
# for i in range(len(n_s)):
#     neighbor[i] = len(adj.getrow(i).indices)
# n_adj = np.zeros(shape=(len(n_s), int(neighbor.mean())))
# for i in range(len(n_s)):
#     n_adj[i] = np.random.choice(adj.getrow(i).indices, n_adj.shape[1], replace=(neighbor[i] < n_adj.shape[1]))
# # n_adj = torch.from_numpy(n_adj).cuda()
# t2 = time.time() - t1
# print(t2)
# from graphsaint.kgraphsaint.dataloader import SubgraphRating
#
# t3 = time.time()
# #   assert all item to train
# item = set(train_data.T[1].tolist())
# assert n_item == len(item)
#
# train_data = train_data.tolist()
# train_data = sorted(train_data, key=lambda key: key[1], reverse=False)
# train_data = np.array(train_data)
# # items = set(train)
# t4 = time.time() - t3
# print(t4)
# # exit(0)
# subgraph = SubgraphRating(n_s, train_data, verbose=True)
# i = 1
