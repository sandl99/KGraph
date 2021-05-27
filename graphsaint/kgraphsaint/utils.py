import numpy as np
from sklearn import metrics
from torch_sparse import SparseTensor
import torch
from torch_scatter import gather_csr
from torch_sparse.storage import SparseStorage, get_layout
from torch_sparse.tensor import SparseTensor
from typing import Union


def build_sample(mini, args):
    train_phases = {
        'sampler': args.sampler,
        'size_subg_edge': args.size_subg_edge
    }
    mini.set_sampler(train_phases)


def reformat_train_ratings(train_data):
    """
        @param train_data: data ratings for train
    """
    train_data = train_data.tolist()
    train_data = sorted(train_data, key=lambda key: key[1], reverse=False)
    return np.array(train_data)


def check_items_train(train_data, n_item):
    item = set(train_data.T[1].tolist())
    assert n_item == len(item)


def auc_score(pred, true, average='micro'):
    return metrics.roc_auc_score(true, pred, average=average)

def f1_score(pred, true):
    return metrics.f1_score(true, pred, average='binary')

def to_dense(src: SparseTensor, dtype) -> torch.Tensor:
    row, col, value = src.coo()

    if value is not None:
        mat = torch.zeros(src.sizes(), dtype=value.dtype,
                            device=src.device())
    else:
        mat = torch.zeros(src.sizes(), dtype=dtype, device=src.device())

    if value is not None:
        mat[row, col] = value
    else:
        mat[row, col] = torch.ones(src.nnz(), dtype=mat.dtype,
                                    device=mat.device)
    return mat


def index_select(src: SparseTensor,
                 idx: Union[torch.Tensor, SparseTensor]) -> SparseTensor:
    if isinstance(idx, SparseTensor):
        idx = idx.to_dense().type(torch.long)

    idx = torch.flatten(idx)

    old_rowptr, col, value = src.csr()
    rowcount = src.storage.rowcount()

    rowcount = rowcount[idx]

    rowptr = col.new_zeros(idx.size(0) + 1)
    torch.cumsum(rowcount, dim=0, out=rowptr[1:])

    row = torch.arange(idx.size(0),
                        device=col.device).repeat_interleave(rowcount)

    perm = torch.arange(row.size(0), device=row.device)
    perm += gather_csr(old_rowptr[idx] - rowptr[:-1], rowptr)

    col = col[perm]

    if value is not None:
        value = value[perm]

    sparse_sizes = (idx.size(0), src.sparse_size(1))

    storage = SparseStorage(row=row, rowptr=rowptr, col=col, value=value,
                            sparse_sizes=sparse_sizes, rowcount=rowcount,
                            colptr=None, colcount=None, csr2csc=None,
                            csc2csr=None, is_sorted=True)
    return src.from_storage(storage)