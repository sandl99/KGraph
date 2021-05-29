import argparse
from functools import reduce
from graphsaint.kgraphsaint import loader, utils
import logging
from graphsaint.kgraphsaint.batchsample import Minibatch
import torch
from graphsaint.kgraphsaint.models import KGraphSAINT
from graphsaint.kgraphsaint.dataloader import SubgraphRating, Rating
from torch.utils.data import DataLoader
import time
from barbar import Bar
import numpy as np

# logging.basicConfig(level=logging.DEBUG)
# torch.autograd.set_detect_anomaly(True)

name_model = ''


parser = argparse.ArgumentParser(description='command line options')
parser.add_argument('--sampler', action="store", dest="sampler", default='node', help="sampler")
parser.add_argument('--lr', action='store', dest='lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--l2_weight', action='store', dest='l2_weight', type=float, default=1e-7, help='l2_weight')
__arg = parser.parse_args()

class Args:
    def __init__(self):
        self.total_train_ins = 0
        # movie
        self.dataset = 'movie'
        self.aggregator = 'sum'
        self.n_epochs = 500
        self.neighbor_sample_size_train = -1
        self.neighbor_sample_size_eval = -1
        self.dim = 32
        self.n_iter = 2
        self.batch_size = 8192 * 8
        self.l2_weight = 1e-7
        self.lr = 2e-2
        self.ratio = 1
        self.save_dir = './kgraph_models'
        self.lr_decay = 0.5
        self.sampler = 'node'
        self.size_subg_edge = 20000
        self.batch_size_eval = 8192 * 8
        # music
        # self.dataset = 'music'
        # self.aggregator = 'sum'
        # self.n_epochs = 500
        # self.neighbor_sample_size_train = -1
        # self.neighbor_sample_size_eval = -1
        # self.dim = 16
        # self.n_iter = 1
        # self.batch_size = 512
        # self.l2_weight = 1e-5
        # self.lr = 1e-3
        # self.ratio = 1
        # self.save_dir = './kgraph_models'
        # self.lr_decay = 0.5
        # self.sampler = 'edge'
        # self.size_subg_edge = 7000
        # self.batch_size_eval = 512

arg = Args()
arg.lr = __arg.lr
arg.l2_weight = __arg.l2_weight
arg.sampler = __arg.sampler

logging.basicConfig(filename=f'./logs/{arg.dataset}/{arg.sampler}_{arg.size_subg_edge}_training.log', filemode='w',
                    format='[%(asctime)s.%(msecs)03d %(filename)s:%(lineno)3s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

phase_iter = 0
print(f'./logs/{arg.dataset}/{arg.sampler}_{arg.size_subg_edge}_training.log')

def parse_arg():
    dcm = Args()
    dcm.lr = __arg.lr
    dcm.l2_weight = __arg.l2_weight
    dcm.sampler = __arg.sampler
    return dcm


def train(_model, _optimizer, _minibatch: Minibatch, _train_data, _device, _args):
    global phase_iter
    phase_iter += 1
    logging.info(f'\n----- Starting training phase {phase_iter} ------ Estimator Epoch: {_minibatch.num_training_batches()}')
    epoch = 0

    while not _minibatch.end():
        _model.train()
        logging.info(f'-------- Mini epoch: {epoch} --------')
        epoch += 1
        _t0 = time.time()
        node, adj = _minibatch.one_batch('train')
        node = torch.from_numpy(node)
        adj = adj.to(_device)
        node = node.to(_device)

        reserve_node = {j.item(): i for i, j in enumerate(node)}

        _t1 = time.time()
        logging.info(f'Sampling sub graph in {_t1-_t0: .3f} seconds')

        subgraph_rating = SubgraphRating(node, _train_data)
        data_loader = DataLoader(subgraph_rating, batch_size=_args.batch_size, drop_last=False, shuffle=True)
        _t2 = time.time()
        logging.info(f'Building DataLoader in {_t2-_t1: .3f} seconds')

        train_loss, eval_loss = 0, 0
        train_auc, eval_auc = [], []
        train_pred, train_true = np.zeros(0), np.zeros(0)
        
        for data in Bar(data_loader):
            users, items, labels = data['user'].to(_device), data['item'].to(_device), data['label'].type(torch.float32).to(_device)
            _optimizer.zero_grad()
            outputs = _model(users, items, reserve_node, node, adj)
            scores_pred = torch.sigmoid(outputs.detach())
            # print(torch.max(outputs), torch.min(outputs))
            norm_loss = _minibatch.norm_loss_train[items]
            bce_weighted_loss = torch.nn.BCEWithLogitsLoss(weight=norm_loss, reduction='sum')
            loss = bce_weighted_loss(outputs, labels)
            loss.backward()
            _optimizer.step()
            train_loss += loss.item()

            # detach outputs and labels
            train_pred = np.concatenate((train_pred, scores_pred.detach().cpu().numpy()))
            train_true = np.concatenate((train_true, labels.detach().cpu().numpy()))
            torch.cuda.empty_cache()
        logging.info(f'Train loss: {train_loss}')
        auc_score = utils.auc_score(train_pred, train_true, 'micro')
        train_pred = np.where(train_pred >= 0.5, 1, 0)
        f1_score = utils.f1_score(train_pred, train_true)
        logging.info(f'Train AUC : {auc_score}')
        logging.info(f'Train F1  : {f1_score}')


def evaluate(_model, _eval_data, _mini_batch: Minibatch, _device, epoch, args):
    global name_model
    node, adj = _mini_batch.one_batch(mode='val')
    node = torch.from_numpy(node).to(_device)
    adj = adj.to(_device)
    eval_loss = 0
    eval_pred, eval_true = np.zeros(0), np.zeros(0)
    data = Rating(_eval_data)
    data_loader = DataLoader(data, batch_size=args.batch_size_eval, shuffle=True)
    _criterion = torch.nn.BCEWithLogitsLoss(reduce='sum')
    _model.eval()

    with torch.no_grad():
        for data in Bar(data_loader):
            users, items, labels = data['user'].to(_device), data['item'].to(_device), data['label'].type(torch.float32).to(_device)
            # _optimizer.zero_grad()
            outputs = _model(users, items, node=node, subgraph=adj)
            loss = _criterion(outputs, labels)
            # loss.backward()
            eval_loss += loss.item()
            # detach outputs and labels
            scores_pred = torch.sigmoid(outputs.detach())
            eval_pred = np.concatenate((eval_pred, scores_pred.detach().cpu().numpy()))
            eval_true = np.concatenate((eval_true, labels. detach().cpu().numpy()))
            torch.cuda.empty_cache()
    eval_loss /= len(data_loader)
    logging.info(f'Eval loss: {eval_loss}')
    auc_score = utils.auc_score(eval_pred, eval_true, 'micro')
    eval_pred = np.where(eval_pred >= 0.5, 1, 0)
    f1_score = utils.f1_score(eval_pred, eval_true)
    logging.info(f'Eval AUC : {auc_score}')
    logging.info(f'Eval F1  : {f1_score}')
    # saving model
    state_dict = {
        'model_state_dict': _model.state_dict(),
        'epoch': epoch,
    }
    state_dict.update(args.__dict__)
    name_model = args.save_dir + '/model_' + args.sampler + '_' + str(args.size_subg_edge) + '_' + str(auc_score) + '_' + str(f1_score) + '.pt'
    torch.save(state_dict, name_model)


def main():
    global name_model
    args = parse_arg()
    for key, item in args.__dict__.items():
        logging.info(f'Parameter {key} := {item}')
    # Loading data
    t0 = time.time()
    logging.info('Loading knowledge graph from data')
    n_entity, n_relation, adj_entity, adj_relation = loader.load_kg(args)
    logging.info('Loading ratings data')
    n_user, n_item, train_data, eval_data, test_data = loader.load_rating(args)
    train_data = utils.reformat_train_ratings(train_data)
    args.total_train_ins = len(train_data)
    t1 = time.time()
    logging.info(f'Done loading data in {t1-t0 :.3f}')

    # Build GraphSAINT sampler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mini_batch = Minibatch(adj_entity, adj_relation, n_entity, n_relation, args)
    utils.build_sample(mini_batch, args)

    # model and optimizer
    model = KGraphSAINT(n_user, n_entity, n_relation, args).to(device)
    logging.info(f'Device: {device}')
    logging.info("Total number of parameters = {}".format(sum(p.numel() for p in model.parameters())))
    epoch = 0
    if name_model != '':
        checkpoint = torch.load(name_model)
        logging.info(f'Loading checkpoint model {name_model}')
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        args.lr = checkpoint['lr']
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_weight)

    # train phases
    for i in range(epoch, 20):
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        logging.debug(f'Training with learning rate {lr}')
        train(model, optimizer, mini_batch, train_data, device, args)
        evaluate(model, eval_data, mini_batch, device, i, args)

        mini_batch.shuffle()
        utils.build_sample(mini_batch, args)


if __name__ == '__main__':
    main()

