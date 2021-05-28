from torch.utils.data import Dataset
import numpy as np
import time
from tqdm import tqdm


def binary_search(arr, k):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] < k:
            low = mid + 1
        elif arr[mid] > k:
            high = mid - 1
        else:
            return True
    return False


def get_missing_items(ratings):
    items = ratings.T[1]
    all_items = set(np.arange(items.max()))
    return all_items - set(items)


def build_ratings(node_subgraph, ratings):
    # create missing items, in movie data, it's empty
    # missing_items = get_missing_items(ratings)
    # tao anh' xa. tu index den item value, anh xa 1 1 neu la du lieu movie
    unique_items = np.unique(ratings.T[1])
    indices = []
    items = ratings.T[1]
    i, j = 0, 0
    while j != len(items):
        if items[i] != items[j]:
            indices.append(i)
            i = j
        j += 1
    indices.append(i)
    indices.append(j)
    items = len(indices)
    remove = []
    for i in range(1, items):
        # if i not in missing_items:
        if not binary_search(node_subgraph, unique_items[i - 1]):
            remove.extend([j for j in range(indices[i - 1], indices[i])])
    # for i in
    dcm = np.delete(ratings, remove, axis=0)
    return dcm


class SubgraphRating(Dataset):
    def __init__(self, node_subgraph, ratings, graph='subgraph', verbose=False):
        for i in range(len(node_subgraph) - 1):
            if node_subgraph[i] > node_subgraph[i+1]:
                assert False
        assert graph == 'subgraph' or graph == 'full'
        self.graph = graph
        t1 = time.time()
        if self.graph == 'subgraph':
            self.ratings = build_ratings(node_subgraph, ratings)
        t2 = time.time()
        if verbose:
            print(f'Building ratings in {t2 - t1}')
            print(f'Number observation: {len(self.ratings)}')

    def __len__(self):
        return self.ratings.shape[0]

    def __getitem__(self, idx):
        return {'user': self.ratings[idx, 0], 'item': self.ratings[idx, 1], 'label': self.ratings[idx, 2]}


class Rating(Dataset):
    def __init__(self, ratings):
        self.ratings = ratings

    def __len__(self):
        return self.ratings.shape[0]

    def __getitem__(self, idx):
        return {'user': self.ratings[idx, 0], 'item': self.ratings[idx, 1], 'label': self.ratings[idx, 2]}
