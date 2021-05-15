import argparse
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

logging.basicConfig(level=logging.DEBUG)
# torch.autograd.set_detect_anomaly(True)

name_model = ''


class Args:
    def __init__(self):
        # movie
        self.dataset = 'movie'
        self.aggregator = 'sum'
        self.n_epochs = 500
        self.neighbor_sample_size_train = 100
        self.neighbor_sample_size_eval = 100
        self.dim = 32
        self.n_iter = 2
        self.batch_size = 256
        self.l2_weight = 1e-7
        self.lr = 2e-2
        self.ratio = 1
        self.save_dir = '../../kgraph_models'
        self.lr_decay = 0.5
        self.sampler = 'node'
        self.size_subg_edge = 8000
        # music
        # self.dataset = 'music'
        # self.aggregator = 'sum'
        # self.n_epochs = 500
        # self.neighbor_sample_size_train = 15
        # self.neighbor_sample_size_eval = 25
        # self.dim = 16
        # self.n_iter = 1
        # self.batch_size = 256
        # self.l2_weight = 1e-4
        # self.lr = 1e-3
        # self.ratio = 1
        # self.save_dir = '../../kgraph_models'
        # self.lr_decay = 0.5
        # self.sampler = 'node'
        # self.size_subg_edge = 8000

arg = Args()
logging.basicConfig(filename=f'./logs/{arg.dataset}/{arg.sampler}_{arg.size_subg_edge}_training.log', filemode='w',
                    format='[%(asctime)s.%(msecs)03d %(filename)s:%(lineno)3s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
phase_iter = 0
print(f'./logs/{arg.dataset}/{arg.sampler}_{arg.size_subg_edge}_training.log')

def parse_arg():
    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
    # parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
    # parser.add_argument('--n_epochs', type=int, default=500, help='the number of epochs')
    # parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
    # parser.add_argument('--dim', type=int, default=16, help='dimension of user and entity embeddings')
    # parser.add_argument('--n_iter', type=int, default=2,
    #                     help='number of iterations when computing entity representation')
    # parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    # parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
    # parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    # parser.add_argument('--ratio', type=float, default=0.8, help='size of training dataset')
    #
    # parser.add_argument('--save_dir', type=str, default='./kgraph_models', help='directory saving KGCN')
    # return parser.parse_args()
    return Args()


def train(_model, _criterion, _optimizer, _minibatch, _train_data, _device, _args):
    global phase_iter
    phase_iter += 1
    logging.info(f'\n----- Starting training phase {phase_iter} ------ Estimator Epoch: {_minibatch.num_training_batches()}')
    epoch = 0

    while not _minibatch.end():
        _model.train()
        logging.info(f'-------- Epoch: {epoch} --------')
        epoch += 1
        _t0 = time.time()
        node, adj, rel = _minibatch.one_batch('train')
        adj = adj.to(_device)
        rel = rel.to(_device)
        node_tensor = torch.from_numpy(np.array(node)).to('cpu')
        reserve_node = {j: i for i, j in enumerate(node)}

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
            outputs = _model(users, items, reserve_node, node_tensor, adj, rel)
            # print(torch.max(outputs), torch.min(outputs))
            loss = _criterion(outputs, labels)
            loss.backward()

            _optimizer.step()
            train_loss += loss.item()

            # detach outputs and labels
            train_pred = np.concatenate((train_pred, outputs.detach().cpu().numpy()))
            train_true = np.concatenate((train_true, labels. detach().cpu().numpy()))
        logging.info(f'Train loss: {train_loss / len(data_loader)}')
        score = utils.auc_score(train_pred, train_true, 'micro')
        logging.info(f'Train AUC : {score} micro')
        # score = utils.auc_score(train_pred, train_true, 'macro')
        # logging.info(f'Train AUC : {score} macro')

        # if epoch == 2:
        #     exit(0)


def evaluate(_model, _criterion, _eval_data, full_adj, full_rel, _device, epoch, args):
    global name_model
    eval_loss = 0
    eval_pred, eval_true = np.zeros(0), np.zeros(0)
    data = Rating(_eval_data)
    data_loader = DataLoader(data, batch_size=args.batch_size, shuffle=True)
    _model.eval()
    for data in Bar(data_loader):
        users, items, labels = data['user'].to(_device), data['item'].to(_device), data['label'].type(torch.float32).to(_device)
        # _optimizer.zero_grad()
        outputs = _model(users, items, adj=full_adj, rel=full_rel, train_mode=False)
        loss = _criterion(outputs, labels)
        loss.backward()
        eval_loss += loss.item()
        # detach outputs and labels
        eval_pred = np.concatenate((eval_pred, outputs.detach().cpu().numpy()))
        eval_true = np.concatenate((eval_true, labels. detach().cpu().numpy()))
    eval_loss /= len(data_loader)
    logging.info(f'Eval loss: {eval_loss}')
    score = utils.auc_score(eval_pred, eval_true, 'micro')
    logging.info(f'Eval AUC : {score} micro')
    # saving model
    state_dict = {
        'model_state_dict': _model.state_dict(),
        'epoch': epoch,
    }
    state_dict.update(args.__dict__)
    name_model = args.save_dir + '/model_' + args.sampler + '_' + str(args.size_budget) + '_' + str(score) + '.pt'
    # torch.save(state_dict, name_model)


def train2(_model, _criterion, _optimizer, _eval_data, full_adj, full_rel, _device, epoch, args):
    eval_loss = 0
    eval_pred, eval_true = np.zeros(0), np.zeros(0)
    data = Rating(_eval_data)
    data_loader = DataLoader(data, batch_size=args.batch_size, shuffle=True)
    _model.train()
    for data in Bar(data_loader):
        users, items, labels = data['user'].to(_device), data['item'].to(_device), data['label'].type(torch.float32).to(_device)
        _optimizer.zero_grad()
        outputs = _model(users, items, adj=full_adj, rel=full_rel, train_mode=False)
        loss = _criterion(outputs, labels)
        loss.backward()
        _optimizer.step()
        eval_loss += loss.item()
        # detach outputs and labels
        eval_pred = np.concatenate((eval_pred, outputs.detach().cpu().numpy()))
        eval_true = np.concatenate((eval_true, labels. detach().cpu().numpy()))
    logging.info(f'Eval loss: {eval_loss / len(data_loader)}')
    score = utils.auc_score(eval_pred, eval_true, 'micro')
    logging.info(f'Eval AUC : {score} micro')


def main():
    global name_model
    args = parse_arg()
    # Loading data
    t0 = time.time()
    logging.info('Loading knowledge graph from data')
    n_entity, n_relation, adj_entity, adj_relation = loader.load_kg(args)
    logging.info('Loading ratings data')
    n_user, n_item, train_data, eval_data, test_data = loader.load_rating(args)
    train_data = utils.reformat_train_ratings(train_data)
    # utils.check_items_train(train_data, n_item)
    t1 = time.time()
    # full_adj, full_rel = loader.load_kg_ver0(args)
    # full_adj, full_rel = torch.from_numpy(full_adj), torch.from_numpy(full_rel)
    logging.info(f'Done loading data in {t1-t0 :.3f}')

    # Build GraphSAINT sampler
    mini_batch = Minibatch(adj_entity, adj_relation, n_entity, n_relation, args)
    utils.build_sample(mini_batch, args)

    # model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    model = KGraphSAINT(n_user, n_entity, n_relation, args).to(device)
    criterion = torch.nn.BCELoss()
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
    for i in range(epoch, 10):
        logging.info(f'------------- EPOCH {i}')
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        logging.debug(f'Training with learning rate {lr}')
        train(model, criterion, optimizer, mini_batch, train_data, device, args)
        # evaluate(model, criterion, eval_data, full_adj, full_rel, device, i, args)
        # if i == 2 or i == 4:
        #     args.lr *= args.lr_decay
        #     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_weight)
        #     logging.debug(f'Learning rate {args.lr}')
        # mini_batch.batch_num = -1


if __name__ == '__main__':
    main()
    # args = Args()
    # print(args.__dict__)
