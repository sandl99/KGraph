from torch_sparse import SparseTensor
import torch
# from torch_sparse import index_select
from torch_scatter import gather_csr
from torch_sparse.storage import SparseStorage, get_layout
from torch_sparse.tensor import SparseTensor


def index_select(src: SparseTensor,
                 idx: torch.Tensor) -> SparseTensor:
    # dim = src.dim() + dim if dim < 0 else dim
    # assert idx.dim() == 1

    # if dim == 0:
    shape = idx.shape
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


def index_select_for_2D_sparse_index(src: SparseTensor, idx: SparseTensor) -> SparseTensor:
    old_row_ptr, col, value = src.csr()
    rowcount = src.storage.rowcount()

    idx_rowcount = idx.storage.rowcount()
    idx = idx.csr()[2]
    rowcount = rowcount[idx]
    rowptr = col.new_zeros(idx.size(0) + 1)
    torch.cumsum(rowcount, dim=0, out=rowptr[1:])

    row = torch.arange(idx.size(0), device=col.device).repeat_interleave(rowcount)
    perm = torch.arange(row.size(0), device=row.device)
    perm += gather_csr(old_row_ptr[idx] - rowptr[:-1], rowptr)

    col = col[perm]

    if value is not None:
        value = value[perm]

    sparse_sizes = (idx.size(0), src.sparse_size(1))

    storage = SparseStorage(row=row, rowptr=rowptr, col=col, value=value,
                            sparse_sizes=sparse_sizes, rowcount=rowcount,
                            colptr=None, colcount=None, csr2csc=None,
                            csc2csr=None, is_sorted=True)
    return src.from_storage(storage)

# build for graph
row = torch.tensor([1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5, 6, 6])
col = torch.tensor([1, 2, 3, 1, 2, 1, 2, 1, 2, 1, 2, 3, 1, 2])
val = torch.tensor([3, 4, 5, 5, 6, 1, 4, 1, 3, 1, 2, 6, 2, 5])

sm = SparseTensor(row=row, col=col, value=val, sparse_sizes=(7,3))
# a = index_select(sm, torch.tensor([[0, 2], [2, 3]]))
# a = 0
a = index_select_for_2D_sparse_index(sm, sm[torch.tensor([0, 1])])