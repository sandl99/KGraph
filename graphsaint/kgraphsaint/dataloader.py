from numpy.core.fromnumeric import swapaxes
from torch.utils.data import Dataset
import numpy as np
import time
from tqdm import tqdm
# from filter import fast_filter

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
    
    # unique_items = np.unique(ratings.T[1])
    # indices = []
    # items = ratings.T[1]
    # print(len(ratings.T[1]) + len(node_subgraph))
    # i, j = 0, 0
    # t0 = time.time()
    # while j != len(items):
    #     if items[i] != items[j]:
    #         indices.append(i)
    #         i = j
    #     j += 1
    # indices.append(i)
    # indices.append(j)
    # items = len(indices)
    # remove = []
    # t1 = time.time()
    # for i in range(1, items):
    #     # if i not in missing_items:
    #     if not binary_search(node_subgraph, unique_items[i - 1]):
    #         remove.extend([j for j in range(indices[i - 1], indices[i])])
    # # for i in
    # t2 = time.time()
    # dcm = np.delete(ratings, remove, axis=0)
    # t3 = time.time()
    # print(f'time: {t1 - t0} {t2 - t1} { t3- t2}')
    # return dcm

    # faster implement
    # mask = np.full(ratings.T[1].shape, False)
    # t0 = time.time()
    element = ratings.T[1]
    mask = test(node_subgraph, element)
    ratings = ratings[mask]
    return ratings

class SubgraphRating(Dataset):
    def __init__(self, node_subgraph, ratings, graph='subgraph', verbose=False):
        # for i in range(len(node_subgraph) - 1):
        #     if node_subgraph[i] > node_subgraph[i+1]:
        #         assert False
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


def test(node_subgraph, ratings):
    mask = [False] * len(ratings)
    node_subgraph = node_subgraph.tolist()
    ratings = ratings.tolist()
    i, j = 0, 0
    print(len(mask) + len(node_subgraph))
    while i != len(mask) and j != len(node_subgraph):
        if ratings[i] == node_subgraph[j]:
            mask[i] = True
            i += 1
        elif ratings[i] > node_subgraph[j]:
            j += 1
        else:
            i += 1
    return mask
# r = [0, 0, 1, 1, 1, 3, 3, 4, 4]
# n = [0, 2, 4, 7, 9]
# print(filter(n, r))