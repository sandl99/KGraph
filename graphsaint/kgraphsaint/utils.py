import numpy as np
from sklearn import metrics


def build_sample(mini, args):
    train_phases = {
        'sampler': args.sampler,
        'size_subg_edge': args.size_budget
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
