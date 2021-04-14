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


def build_ratings(node_subgraph, ratings):
    indices = []
    items = ratings.T[1]
    i, j = 0, 0
    while j != len(items):
        if items[i] != items[j]:
            indices.append(i)
            i = j
        j += 1
    indices.append(j)
    items = len(indices) - 1
    remove = []
    for i in range(items):
        if not binary_search(node_subgraph, i):
            remove.extend([j for j in range(indices[i], indices[i + 1])])
    dcm = np.delete(ratings, remove, axis=0)
    return dcm


class SubgraphRating(Dataset):
    def __init__(self, node_subgraph, ratings, graph='subgraph', verbose=False):
        for i in range(len(node_subgraph) - 1):
            if node_subgraph[i] > node_subgraph[i+1]:
                assert 1 == 2
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

