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
from torch_sparse import SparseTensor
import torch


def adj_norm(adj, deg=None, sort_indices=True):
    """
    Normalize adj according to the method of rw normalization.
    Note that sym norm is used in the original GCN paper (kipf),
    while rw norm is used in GraphSAGE and some other variants.
    Here we don't perform sym norm since it doesn't seem to
    help with accuracy improvement.

    # Procedure:
    #       1. adj add self-connection --> adj'
    #       2. D' deg matrix from adj'
    #       3. norm by D^{-1} x adj'
    if sort_indices is True, we re-sort the indices of the returned adj
    Note that after 'dot' the indices of a node would be in descending order
    rather than ascending order
    """
    diag_shape = (adj.shape[0],adj.shape[1])
    D = adj.sum(1).flatten() if deg is None else deg
    norm_diag = sp.dia_matrix((1/D,0),shape=diag_shape)
    adj_norm = norm_diag.dot(adj)
    if sort_indices:
        adj_norm.sort_indices()
    return adj_norm


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
        # remove relation data
        value = np.ones(value.shape)
        return sp.csr_matrix((value, (row, col)), shape=(n_ent, n_ent))


def build_adj_matrix(node, csr, neighbor_size=50):
    # assert node[0] != 0
    # neighbor = np.zeros(shape=len(node) + 1, dtype=np.int64)
    # for i in range(1, neighbor.shape[0]):
    #     neighbor[i] = len(csr.getrow(i - 1).indices)
    # n_adj = np.zeros(shape=(neighbor.shape[0], neighbor_size), dtype=np.int64)
    # for i in range(1, neighbor.shape[0]):
    #     if neighbor[i] < neighbor_size:
    #         n_adj[i] = np.append(csr.getrow(i - 1).indices, np.zeros(shape=neighbor_size - neighbor[i]))
    #     else:
    #         n_adj[i] = np.random.choice(csr.getrow(i - 1).indices, n_adj.shape[1], replace=False)
    # return n_adj
    neighbor = torch.zeros(len(node) + 1, dtype=torch.long)
    zero = torch.zeros(1, dtype=torch.long)
    if neighbor_size != -1:
        for i in range(1, neighbor.size(0)):
            neighbor[i] = min(len(csr.getrow(i - 1).indices), neighbor_size)
    else:
        for i in range(1, neighbor.size(0)):
            neighbor[i] = len(csr.getrow(i - 1).indices)
        neighbor_size = neighbor.max()
    rowptr = torch.cumsum(torch.cat((zero, neighbor), dim=0), dim=0)
    col = [torch.arange(i, dtype=torch.long)[:neighbor_size] for i in neighbor]
    val = [torch.from_numpy(csr.getrow(i - 1).indices[:neighbor_size]) for i in range(1, neighbor.size(0))]
    
    col = torch.cat(col, dim=0)
    val = torch.cat(val, dim=0)
    return SparseTensor(rowptr=rowptr, col=col, value=val, sparse_sizes=(neighbor.size(0), neighbor_size))



def build_rel_matrix(node, csr, adj: SparseTensor):
    rowptr = adj.storage.rowptr().detach()
    col = adj.storage.col().detach()
    neighbor_size = adj.storage.sparse_sizes()[1]
    val = [torch.from_numpy(csr.getrow(i - 1).data[:neighbor_size]) for i in range(1,adj.storage.sparse_sizes()[0] )]
    val = torch.cat(val, dim=0)
    return SparseTensor(rowptr=rowptr, col=col, value=val, sparse_sizes=adj.storage.sparse_sizes())

# def build_edge_index_matrix():


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

    def __init__(self, adj_entity, adj_relation, n_entity, n_relation, args, is_cuda=True):
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.adj_full = _adj_to_sparse_matrix(adj_entity, adj_relation, n_entity, 'csr')
        self.is_cuda = is_cuda
        if self.is_cuda:
            if not torch.cuda.is_available():
                self.is_cuda = False

        self.node_subgraph = None
        self.batch_num = -1
        self.total_node = n_entity

        self.method_sample = None
        self.subgraphs_remaining_indptr = []
        self.subgraphs_remaining_indices = []
        self.subgraphs_remaining_data = []
        self.subgraphs_remaining_nodes = []
        self.subgraphs_remaining_edge_index = []
        
        tmp1, tmp2 = self.adj_full.nonzero()
        value, count = np.unique(tmp1, return_counts=True)
        self.deg_train = count
        self.args = args
        self.sample_coverage = 50
        self.norm_loss_train = np.zeros(self.adj_full.shape[0])
        self.norm_aggr_train = np.zeros(self.adj_full.size)
        self.node_for_sampler  = value
        # normalize for adj_full
        degree = np.repeat(self.deg_train, self.deg_train)
        value = self.adj_full.data / degree
        coo_matrix = self.adj_full.tocoo()
        rowcol = torch.cat((torch.from_numpy(coo_matrix.row).unsqueeze(0), torch.from_numpy(coo_matrix.col).unsqueeze(0)), dim=0)
        self.adj_norm_full = torch.sparse_coo_tensor(rowcol, torch.from_numpy(value.astype(np.float32)), coo_matrix.shape)
        self.adj_norm_full = self.adj_norm_full.coalesce()
        i = 1

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
        # check in 27/5/2021
        # indptr = self.adj_full.indptr
        # indices = self.adj_full.indices
        # cnt = 0
        # self.reserve_edge = dict()
        # for i in range(len(indptr) - 1):
        #     for j in range(indptr[i], indptr[i + 1]):
        #         self.reserve_edge[(i, indices[j])] = cnt
        #         cnt += 1

        if self.method_sample == 'mrw':
            raise NotImplementedError
        elif self.method_sample == 'rw':
            self.size_subg_budget = 3000 * 2 # train_phases['size_subg_edge']
            self.graph_sampler = rw_sampling(
                self.adj_full,
                self.node_for_sampler,
                self.size_subg_budget,
                3000,
                2,
            )
        elif self.method_sample == 'node':
            self.size_subg_budget = train_phases['size_subg_edge']
            self.graph_sampler = node_sampling(
                self.adj_full,
                self.node_for_sampler,
                train_phases['size_subg_edge']
            )
        elif self.method_sample == 'edge':
            self.size_subg_budget = train_phases['size_subg_edge'] * 2
            self.graph_sampler = edge_sampling(
                self.adj_full,
                self.node_for_sampler,
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
        while True:
            self.par_graph_sample('train')
            tot_sampled_nodes = sum([len(n) for n in self.subgraphs_remaining_nodes])
            if tot_sampled_nodes > self.sample_coverage * self.adj_full.shape[0]:
                break
        print()
        num_subg = len(self.subgraphs_remaining_nodes)
        for i in range(num_subg):
            self.norm_aggr_train[self.subgraphs_remaining_edge_index[i]] += 1
            self.norm_loss_train[self.subgraphs_remaining_nodes[i]] += 1
        # assert self.norm_loss_train[self.node_val].sum() + self.norm_loss_train[self.node_test].sum() == 0
        for v in range(self.adj_full.shape[0]):
            i_s = self.adj_full.indptr[v]
            i_e = self.adj_full.indptr[v + 1]
            if i_s != i_e:
                val = np.clip(self.norm_loss_train[v] / self.norm_aggr_train[i_s : i_e], 0, 1e4)
                val[np.isnan(val)] = 0.1
                self.norm_aggr_train[i_s : i_e] = val
        self.norm_loss_train[np.where(self.norm_loss_train==0)[0]] = 0.1
        # self.norm_loss_train[self.node_val] = 0
        # self.norm_loss_train[self.node_test] = 0
        self.norm_loss_train = num_subg / self.norm_loss_train / self.adj_full.shape[0]
        self.norm_loss_train = self.norm_loss_train.astype(np.float32)
        if self.is_cuda:
            self.norm_loss_train = torch.from_numpy(self.norm_loss_train).cuda()
            # self.norm_aggr_train = torch.from_numpy(self.norm_aggr_train).cuda()

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
            self.node_subgraph = np.arange(self.adj_norm_full.shape[0])
            adj = self.adj_norm_full
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
            # t_row = [self.node_subgraph[i] for i in row]
            # t_col = [self.node_subgraph[i] for i in col]
            # data = [self.adj_full[i, j] for i, j in zip(t_row, t_col)]
            # adj = sp.csr_matrix((data, (row, col)), shape=(self.size_subgraph, self.size_subgraph))
            # adj = SparseTensor(row=torch.tensor(t_row), col=torch.tensor(t_col), value=torch.tensor(data), sparse_sizes=(self.size_subgraph + 1, self.size_subgraph + 1))
            adj_edge_index = self.subgraphs_remaining_edge_index.pop()
            adj_edge_index = self.norm_aggr_train[adj_edge_index]

            degree = self.deg_train[self.node_subgraph]
            repeat = np.array([indptr[i+1] - indptr[i] for i in range(len(indptr) - 1)])
            degree = np.repeat(degree, repeat)
            adj_edge_index /= degree
            indices = np.concatenate((np.array(row).reshape(1, -1), col.reshape(1, -1)))
            adj = torch.sparse_coo_tensor(torch.from_numpy(indices), torch.from_numpy(adj_edge_index), (self.size_subgraph, self.size_subgraph))
            adj = adj.coalesce()
            print("{} nodes, {} edges".format(self.node_subgraph.size, indices.shape[1]))
            self.batch_num += 1
        return self.node_subgraph, adj

    def num_training_batches(self):
        return math.ceil(self.node_for_sampler.shape[0] / float(self.size_subg_budget))

    def shuffle(self):
        self.node_for_sampler = np.random.permutation(self.node_for_sampler)
        self.batch_num = -1

    def end(self):
        return (self.batch_num + 1) * self.size_subg_budget >= self.node_for_sampler.shape[0]

