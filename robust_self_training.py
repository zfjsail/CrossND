import argparse
from os.path import join
import os
import time
from datetime import datetime
import json
from copy import deepcopy
import numpy as np
from collections import defaultdict as dd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

from data_loader import AuthorPaperPairMatchDataset, AuthorPaperTriIcsEvalDataset
from model import CroNDBase
import utils
from utils import ChunkSampler
import settings

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--matrix-size1', type=int, default=settings.MAX_MAT_SIZE, help='Matrix size 1.')
parser.add_argument('--n-papers-per-author', type=int, default=10, help='Matrix size 1.')
parser.add_argument('--mat1-channel1', type=int, default=20, help='Matrix1 number of channels1.')
parser.add_argument('--mat1-hidden', type=int, default=64, help='Matrix1 hidden dim.')  # matters 8 is poor
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--weight-decay', type=float, default=1e-2, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--check-point', type=int, default=5, help="Check point")
parser.add_argument('--shuffle', action='store_true', default=True, help="Shuffle dataset")
parser.add_argument('--file-dir', type=str, default=settings.OUT_DATASET_DIR, help="Input file directory")
parser.add_argument('--debug-scale', type=int, default=10000, help="Training ratio (0, 100)")
parser.add_argument('--forget-rate', type=float, default=0.1, help='Forget rate.')

parser.add_argument('--lr', type=float, default=1e-5, help='Initial learning rate.')
parser.add_argument('--epsilon', type=float, default=0.6, help='Logit weight for soft labeling.')
parser.add_argument('--batch', type=int, default=4096, help="Batch size")
parser.add_argument('--loss-type', type=str, default="ce", help="Loss type")
parser.add_argument('--conv-type', type=str, default="kc", help="Conv type")
parser.add_argument('--conv-tune', type=str, default="yes", help="Tune conv paras")
parser.add_argument('--suffix', type=str, default="None", help="Suffix")
parser.add_argument('--esb', type=bool, default=True, help="Whether to ensemble")
parser.add_argument('--perfect-data', type=bool, default=False, help="Training ratio (0, 100)")
parser.add_argument('--debug', type=bool, default=False, help="Training ratio (0, 100)")
parser.add_argument('--mat1-kernel-size1', type=int, default=4, help='Matrix1 kernel size1.')
args = parser.parse_args()

# post-process config -- not for tuning
if settings.data_source == "aminer":
    args.mat1_kernel_size1 = 4
    # args.conv_type = "cnn"
    # args.conv_tune = "no"
    args.suffix = "pctpospmin03negpmax009"
elif settings.data_source == "kddcup":
    args.mat1_kernel_size1 = 3
    args.batch = 256
else:
    raise NotImplementedError


# ics_triplets_all = utils.load_json(settings.DATASET_DIR, "ics_triplets_via_author_sim.json")
# ics_triplets_valid = utils.load_json(settings.DATASET_DIR, "ics_triplets_relabel1_valid.json")
# ics_triplets_test = utils.load_json(settings.DATASET_DIR, "ics_triplets_relabel1_test.json")

# ics_pid_idx_keys = {x["pid"] for x in ics_triplets_all}


def forget_rate_scheduler(epochs, forget_rate, num_gradual, exponent):
    """Tells Co-Teaching what fraction of examples to forget at each epoch."""
    # define how many things to forget at each rate schedule
    forget_rate_schedule = np.ones(epochs) * forget_rate
    forget_rate_schedule[:num_gradual] = np.linspace(
        0, forget_rate ** exponent, num_gradual)
    return forget_rate_schedule


def loss_coteaching(y_1, y_2, t, forget_rate):
    loss_1 = F.cross_entropy(y_1, t, reduction='none')
    ind_1_sorted = np.argsort(loss_1.data.cpu())
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduction='none')
    ind_2_sorted = np.argsort(loss_2.data.cpu())

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]

    # Share updates between the two models.
    # TODO: these class weights should take into account the ind_mask filters.
    loss_1_update = F.cross_entropy(
        y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(
        y_2[ind_1_update], t[ind_1_update])

    return (
        torch.sum(loss_1_update) / num_remember,
        torch.sum(loss_2_update) / num_remember,
    )


def loss_cross_entropy(epoch, y, t, ind, loss_all, class_weight):
    # Record loss and loss_div for further analysis
    loss = F.cross_entropy(y, t, class_weight, reduction='none')
    loss_numpy = loss.data.cpu().numpy()
    num_batch = len(loss_numpy)
    loss_all[ind, epoch] = loss_numpy
    return torch.sum(loss) / num_batch


def f_beta(epoch):
    beta1 = np.linspace(0.0, 0.0, num=10)
    beta2 = np.linspace(0.0, 2, num=30)
    beta3 = np.linspace(2, 2, num=60)

    beta = np.concatenate((beta1, beta2, beta3), axis=0)
    return beta[epoch]


def loss_cores(epoch, y, t, ind, loss_all, loss_div_all, noise_prior=None):
    beta = f_beta(epoch)
    # if epoch == 1:
    #     print(f'current beta is {beta}')
    loss = F.cross_entropy(y, t, reduction='none')
    loss_numpy = loss.data.cpu().numpy()
    num_batch = len(loss_numpy)
    loss_v = np.zeros(num_batch)
    loss_div_numpy = float(np.array(0))
    loss_ = -torch.log(F.softmax(y, dim=1) + 1e-8)
    # sel metric
    loss_sel = loss - torch.mean(loss_, 1)
    if noise_prior is None:
        loss = loss - beta * torch.mean(loss_, 1)
    else:
        loss = loss - beta * torch.sum(torch.mul(noise_prior, loss_), 1)

    loss_div_numpy = loss_sel.data.cpu().numpy()
    loss_all[ind, epoch] = loss_numpy
    loss_div_all[ind, epoch] = loss_div_numpy
    for i in range(len(loss_numpy)):
        if epoch <= 30:
            loss_v[i] = 1.0
        elif loss_div_numpy[i] <= 0:
            loss_v[i] = 1.0
    loss_v = loss_v.astype(np.float32)
    loss_v_var = Variable(torch.from_numpy(loss_v)).cuda()
    loss_ = loss_v_var * loss
    if sum(loss_v) == 0.0:
        return torch.mean(loss_) / 100000000
    else:
        return torch.sum(loss_) / sum(loss_v), loss_v.astype(int)


def label_correction_loss(logits, label, epsilon=args.epsilon):  # whoiswho old & kddcup now
    label = label.unsqueeze(1)
    label = torch.cat((1 - label, label), dim=1)
    new_target_probs = (1 - epsilon) * label + epsilon * logits
    num_examples = logits.shape[0]
    loss = torch.sum(new_target_probs * (-torch.log(logits + 1e-6)), 1)
    loss = sum(loss) / num_examples
    return loss


def dev_simple_on_ics(data_loader, model, role):
    model.eval()
    labels = []
    scores = []

    for dev_batch in data_loader:
        x, x_stat, Y = dev_batch

        if args.cuda:
            x = x.cuda()
            Y = Y.cuda()
            x_stat = x_stat.cuda()

        with torch.no_grad():
            batch_score, _ = model(x.float(), x_stat.float())
            batch_score_0 = torch.exp(batch_score[:, 0])
            batch_score_0 = batch_score_0.detach().cpu().tolist()
            scores += batch_score_0
            labels += [1 - x for x in Y.data.cpu().tolist()]

    # if role == "valid":
    #     ics_eval_triplets = ics_triplets_valid
    # elif role == "test":
    #     ics_eval_triplets = ics_triplets_test
    # else:
    #     raise NotImplementedError

    # for i in range(len(ics_eval_triplets)):
    #     cur_pid = ics_eval_triplets[i]["pid"]
    #     if cur_pid in ics_pid_idx_keys:
    #         scores[i] = 1

    auc = roc_auc_score(labels, scores)
    maps = average_precision_score(labels, scores)
    model.train()
    return auc, maps


def dev(args, model, dev_loader, pairs=None):
    model.eval()
    labels = []
    scores = []
    scores_1 = []
    loss = 0.
    total = 0.
    aid_to_label = dd(list)
    aid_to_score = dd(list)

    i = 0
    for dev_batch in dev_loader:
        x, Y, f_add, _ = dev_batch
        bs = Y.shape[0]

        if args.cuda:
            x = x.cuda()
            Y = Y.cuda()
            f_add = f_add.cuda()

        with torch.no_grad():
            batch_score, _ = model(x.float(), f_add.float())
            batch_score_0 = torch.exp(batch_score[:, 0])
            batch_score_1 = torch.exp(batch_score[:, 1])
            batch_score_0 = batch_score_0.detach().cpu().tolist()
            batch_score_1 = batch_score_1.detach().cpu().tolist()
            labels += [1 - x for x in Y.data.cpu().tolist()]
            scores += batch_score_0
            scores_1 += batch_score_1
            cur_loss = F.nll_loss(batch_score, Y)
            loss += bs * cur_loss
            total += bs

        for j in range(bs):
            cur_pair = pairs[i + j]
            cur_aid = cur_pair["aid1"]
            cur_pid = cur_pair["pid"]
            aid_to_label[cur_aid].append(1 - cur_pair["label"])
            aid_to_score[cur_aid].append(scores[i + j])
            # if cur_pid not in ics_pid_idx_keys:
            #     aid_to_score[cur_aid].append(scores[i + j])
            # else:
            #     aid_to_score[cur_aid].append(1)

        i += bs

    map_sum = 0
    map_weight = 0
    auc_sum = 0
    n_authors = 0
    for aid in aid_to_label:
        cur_labels = aid_to_label[aid]
        if sum(cur_labels) / len(cur_labels) >= 0.5 or sum(cur_labels) == 0:
            continue
        n_authors += 1
        cur_map = average_precision_score(aid_to_label[aid], aid_to_score[aid])
        map_sum += cur_map / len(aid_to_label[aid])
        map_weight += 1 / len(aid_to_label[aid])
        cur_auc = roc_auc_score(aid_to_label[aid], aid_to_score[aid])
        auc_sum += cur_auc / len(aid_to_label[aid])

    map_avg = map_sum / map_weight
    auc_avg = auc_sum / map_weight

    model.train()
    return auc_avg, map_avg, loss / total


def eval_per_person_nd_results(args=args):
    now = datetime.now()
    now = now.strftime("%Y%m%d")
    args.file_dir = settings.OUT_DATASET_DIR
    model_dir = join(args.file_dir, "models", now)

    test_pairs = utils.load_json(settings.DATASET_DIR, "eval_na_checking_triplets_relabel1_test.json")

    dataset = AuthorPaperPairMatchDataset(args.file_dir, args.matrix_size1, args.n_papers_per_author, args.seed,
                                            shuffle=False, role="test", args=args)
    N = len(dataset)

    test_loader = DataLoader(dataset, batch_size=args.batch, sampler=ChunkSampler(N, 0))

    cur_loader = test_loader
    cur_pairs = test_pairs

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    logger.info('cuda is available %s', args.cuda)

    out_dir = join(settings.RESULT_DIR, "self-ablation", now)

    for i in range(0, 5):
        cur_seed = args.seed + i
        logger.info("cur_seed %s", cur_seed)
        model = CroNDBase(in_channels=args.n_papers_per_author, matrix_size=args.matrix_size1, channel1=args.mat1_channel1, 
                          kernel_size1=args.mat1_kernel_size1, hidden_size=args.mat1_hidden, conv_type=args.conv_type)
        model = model.float()
        model.load_state_dict(torch.load(join(model_dir, 'paper_author_crond_base_matching_{}.mdl'.format(cur_seed))))
        model.eval()
        model.cuda()

        aid_to_label = dd(list)
        aid_to_score = dd(list)
        aid_to_name = {}
        aid_to_metrics = {}

        labels = []
        scores = []
        scores_1 = []
        loss = 0.
        total = 0.

        i = 0
        for dev_batch in cur_loader:
            x, Y, f_add, _ = dev_batch
            bs = Y.shape[0]

            if args.cuda:
                x = x.cuda()
                Y = Y.cuda()
                f_add = f_add.cuda()

            with torch.no_grad():
                batch_score, _ = model(x.float(), f_add.float())
                batch_score_0 = torch.exp(batch_score[:, 0])
                batch_score_1 = torch.exp(batch_score[:, 1])
                batch_score_0 = batch_score_0.detach().cpu().tolist()
                batch_score_1 = batch_score_1.detach().cpu().tolist()
                labels += [1 - x for x in Y.data.cpu().tolist()]
                scores += batch_score_0
                scores_1 += batch_score_1
                cur_loss = F.nll_loss(batch_score, Y)
                loss += bs * cur_loss
                total += bs

            for j in range(bs):
                cur_pair = cur_pairs[i + j]
                cur_aid = cur_pair["aid1"]
                aid_to_name[cur_aid] = cur_pair["name"]
                aid_to_label[cur_aid].append(1 - cur_pair["label"])
                aid_to_score[cur_aid].append(scores[i + j])
            i += bs
        
        for aid in aid_to_label:
            cur_labels = aid_to_label[aid]
            if sum(cur_labels) / len(cur_labels) >= 0.5 or sum(cur_labels) == 0:
                continue
            cur_map = average_precision_score(aid_to_label[aid], aid_to_score[aid])
            cur_auc = roc_auc_score(aid_to_label[aid], aid_to_score[aid])
            aid_to_metrics[aid] = (cur_auc, cur_map)
        
        with open(join(out_dir, "results_per_name_seed_{}.json".format(cur_seed)), "w") as wf:
            aids = sorted(aid_to_metrics.keys())
            for aid in aids:
                wf.write("{}\t{}\t{:.4f}\t{:.4f}\n".format(aid, aid_to_name[aid], aid_to_metrics[aid][0], aid_to_metrics[aid][1]))
                wf.flush()


def train(epoch, train_loader, model, model2, optimizer, optimizer2, forget_rate_schedule, loss_all, loss_div_all, class_weight, args):
    model.train()
    if args.loss_type == "ct":
        model2.train()

    loss = 0.
    total = 0.

    loss2 = 0.
    loss_2 = 0.

    v_list = np.zeros(len(loss_all))
    idx_each_class_noisy = [[] for i in range(2)]
    loss_mae = nn.L1Loss()

    for i_batch, batch in enumerate(train_loader):
        x1, Y, f_add, idx = batch
        bs = Y.shape[0]

        if args.cuda:
            x1 = x1.cuda()
            Y = Y.cuda()
            f_add = f_add.cuda()

        optimizer.zero_grad()
        output, out_linear = model(x1.float(), f_add.float())
        if args.loss_type == "ct":
            output2, out_linear2 = model2(x1.float(), f_add.float())
        else:
            out_linear2 = None

        if args.loss_type == "ce":
            loss_train = loss_cross_entropy(epoch, out_linear, Y, idx, loss_all, class_weight)
        elif args.loss_type == "cores":
            loss_train, loss_v = loss_cores(epoch, out_linear, Y, idx, loss_all, loss_div_all)
            v_list[idx] = loss_v
            for i in range(bs):
                if loss_v[i] == 0:
                    idx_each_class_noisy[Y[i]].append(idx[i])
        elif args.loss_type == "ls":  # label smoothing
            loss_train = label_correction_loss(torch.exp(output), Y)
        elif args.loss_type == "mae":
            loss_train = loss_mae(torch.exp(output)[:, 1], Y)
        elif args.loss_type == "ct":
            loss_train, loss_2 = loss_coteaching(out_linear, out_linear2, Y,
                                                 forget_rate=forget_rate_schedule[epoch])
        else:
            print('loss type not supported')
            raise SystemExit

        loss += bs * loss_train.item()
        total += bs
        loss_train.backward()
        optimizer.step()

        if args.loss_type == "ct":
            loss2 += bs * loss_2.item()
            optimizer2.zero_grad()
            loss_2.backward()
            optimizer2.step()

    logger.info("train loss epoch %d: %f, %f", epoch, loss / total, loss2 / total)


def main(args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    logger.info('cuda is available %s', args.cuda)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if args.debug:
        args.file_dir = join(args.file_dir, "debug-{}".format(args.debug_scale))
    else:
        args.debug_scale = 10000000
    print("file dir", args.file_dir)

    dataset = AuthorPaperPairMatchDataset(args.file_dir, args.matrix_size1, args.n_papers_per_author, args.seed,
                                          args.shuffle, args=args, perfect_data=args.perfect_data)
    N = len(dataset)

    class_weight = dataset.class_weight
    logger.info("********************************************************")
    logger.info("class_weight=%.2f:%.2f", class_weight[0], class_weight[1])

    train_loader = DataLoader(dataset, batch_size=args.batch,
                              sampler=ChunkSampler(N, 0))

    loss_all = np.zeros((N, args.epochs))
    loss_div_all = np.zeros((N, args.epochs))

    model = CroNDBase(in_channels=args.n_papers_per_author, matrix_size=args.matrix_size1, channel1=args.mat1_channel1,
                      kernel_size1=args.mat1_kernel_size1, hidden_size=args.mat1_hidden, conv_type=args.conv_type,
                      esb=args.esb, conv_tune=args.conv_tune == "yes")

    model = model.float()

    if args.loss_type == "ct":
        model2 = CroNDBase(in_channels=args.n_papers_per_author, matrix_size=args.matrix_size1,
                           channel1=args.mat1_channel1, kernel_size1=args.mat1_kernel_size1,
                           hidden_size=args.mat1_hidden, conv_type=args.conv_type, conv_tune=args.conv_tune == "yes")
        model2 = model2.float()
        if args.cuda:
            model2.cuda()
        optimizer2 = optim.AdamW(model2.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        model2 = None
        optimizer2 = None

    if args.cuda:
        model.cuda()
        class_weight = class_weight.cuda()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    dataset = AuthorPaperPairMatchDataset(args.file_dir, args.matrix_size1, args.n_papers_per_author, args.seed,
                                          shuffle=False, role="valid", args=args)
    N = len(dataset)

    valid_loader2 = DataLoader(dataset, batch_size=args.batch, sampler=ChunkSampler(N, 0))

    dataset = AuthorPaperPairMatchDataset(args.file_dir, args.matrix_size1, args.n_papers_per_author, args.seed,
                                          shuffle=False, role="test", args=args)
    N = len(dataset)

    test_loader2 = DataLoader(dataset, batch_size=args.batch, sampler=ChunkSampler(N, 0))

    dataset = AuthorPaperTriIcsEvalDataset(args.file_dir, args.matrix_size1, args.n_papers_per_author, args,
                                           role="valid", suffix=args.suffix)
    N = len(dataset)
    valid_loader_ics = DataLoader(dataset, batch_size=args.batch, sampler=ChunkSampler(N, 0))

    dataset = AuthorPaperTriIcsEvalDataset(args.file_dir, args.matrix_size1, args.n_papers_per_author, args,
                                           role="test", suffix=args.suffix)
    N = len(dataset)
    test_loader_ics = DataLoader(dataset, batch_size=args.batch, sampler=ChunkSampler(N, 0))

    t_total = time.time()
    logger.info("training...")

    auc_max_valid = 0
    map_max_valid = 0
    map_max_final = 0
    best_model = None
    best_round = -1
    auc_test_final = 0
    best_metric_valid = 0

    auc_ics_best_valid = 0
    map_ics_best_valid = 0
    auc_ics_best_test = 0
    map_ics_best_test = 0

    forget_rate_schedule = forget_rate_scheduler(args.epochs, args.forget_rate, int(args.epochs * 3 / 10), 2)

    valid_pairs = utils.load_json(settings.DATASET_DIR, "eval_na_checking_triplets_relabel1_valid.json")
    test_pairs = utils.load_json(settings.DATASET_DIR, "eval_na_checking_triplets_relabel1_test.json")

    now = datetime.now()
    now = now.strftime("%Y%m%d")
    out_dir = join(settings.OUT_DIR, settings.data_source, now)
    os.makedirs(out_dir, exist_ok=True)

    wf = open(join(out_dir, "{}_robust_self_{}_seed_{}_suffix_{}_epoch_metrics.txt".format(
        settings.data_source, args.loss_type, args.seed, args.suffix)), "w")

    cur_auc_ics_test, cur_map_ics_test = dev_simple_on_ics(test_loader_ics, model, role="test")
    wf.write("{:.6f},{:.6f}\n".format(cur_auc_ics_test, cur_map_ics_test))
    wf.flush()

    for epoch in range(args.epochs):
        train(epoch, train_loader, model, model2, optimizer, optimizer2, forget_rate_schedule, loss_all, loss_div_all, class_weight, args=args)

        if (epoch + 1) % args.check_point == 0:
            cur_auc_valid, cur_map_valid, loss_valid = dev(args, model, valid_loader2, pairs=valid_pairs)
            cur_auc_test, cur_map_test, loss_test = dev(args, model, test_loader2, pairs=test_pairs)

            cur_auc_ics_valid, cur_map_ics_valid = dev_simple_on_ics(valid_loader_ics, model, role="valid")
            cur_auc_ics_test, cur_map_ics_test = dev_simple_on_ics(test_loader_ics, model, role="test")

            logger.info("round %d, valid auc %.4f, valid map %.4f", epoch, cur_auc_valid, cur_map_valid)
            logger.info("round %d, test auc %.4f, test map %.4f", epoch, cur_auc_test, cur_map_test)
            logger.info("round %d, valid auc ics %.4f, valid map ics %.4f", epoch, cur_auc_ics_valid, cur_map_ics_valid)
            logger.info("round %d, test auc ics %.4f, test map ics %.4f", epoch, cur_auc_ics_test, cur_map_ics_test)
            logger.info("valid loss %.4f, test loss %.4f", loss_valid, loss_test)

            wf.write("{:.6f},{:.6f}\n".format(cur_auc_ics_test, cur_map_ics_test))
            wf.flush()

            if best_metric_valid < cur_auc_valid + cur_map_valid:
                best_round = epoch
                best_model = deepcopy(model)
                auc_test_final = cur_auc_test
                auc_max_valid = cur_auc_valid
                map_max_valid = cur_map_valid
                map_max_final = cur_map_test
                best_metric_valid = cur_auc_valid + cur_map_valid

                auc_ics_best_valid = cur_auc_ics_valid
                map_ics_best_valid = cur_map_ics_valid
                auc_ics_best_test = cur_auc_ics_test
                map_ics_best_test = cur_map_ics_test

            logger.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            logger.info("best metric valid %.4f", best_metric_valid)
            logger.info("best round %d, best auc valid now %.4f, best map valid now %.4f", best_round, auc_max_valid,
                        map_max_valid)
            logger.info("best... test auc %.4f, test map %.4f", auc_test_final, map_max_final)
            logger.info("br best auc ics valid %.4f, best map ics valid %.4f", auc_ics_best_valid, map_ics_best_valid)
            logger.info("br best auc ics test %.4f, best map ics test %.4f", auc_ics_best_test, map_ics_best_test)

    logger.info("optimization Finished!")
    logger.info("total time elapsed: {:.4f}s".format(time.time() - t_total))

    wf.close()

    now = datetime.now()
    now = now.strftime("%Y%m%d")

    model_dir = join(args.file_dir, "models", now)
    os.makedirs(model_dir, exist_ok=True)
    torch.save(best_model.state_dict(), join(model_dir, 'paper_author_crond_base_matching_{}.mdl'.format(args.seed)))
    logger.info('paper author matching model saved')

    auc_best_model_valid, map_best_model_valid, loss_valid = dev(args, best_model, valid_loader2, pairs=valid_pairs)
    auc_best_model_test, map_best_model_test, loss_test = dev(args, best_model, test_loader2, pairs=test_pairs)
    auc_ics_best_test, map_ics_best_test = dev_simple_on_ics(test_loader_ics, best_model, role="test")

    logger.info("best metric valid %.4f", best_metric_valid)
    logger.info("Finally, best round %d\n*****best auc valid now %.4f, best map valid now %.4f\n"
                "*****best auc test %.4f, best map test now %.4f",
                best_round, auc_best_model_valid, map_best_model_valid, auc_best_model_test, map_best_model_test)
    logger.info("*****best auc ics test %.4f, best map ics test %.4f", auc_ics_best_test, map_ics_best_test)

    return auc_best_model_test, map_best_model_test, auc_ics_best_test, map_ics_best_test, \
           auc_best_model_valid, map_best_model_valid, best_round, auc_best_model_valid + map_best_model_valid


def multiple_robust_train(n_runs=5, args=args, comment="None"):
    now = datetime.now()
    now = now.strftime("%Y%m%d")
    out_dir = join(settings.RESULT_DIR, "self-ablation", now)
    os.makedirs(out_dir, exist_ok=True)
    wf = open(join(out_dir, "{}_robust_self_{}_lr_{}_kernelsize_{}_esb_{}_epsilon_{}_convtype_{}_convtune_{}_perfectdata_{}_suffix_{}_runs_{}_{}.txt".format(
        settings.data_source, args.loss_type, args.lr, args.mat1_kernel_size1, args.esb, args.epsilon, args.conv_type, args.conv_tune,
        args.perfect_data, args.suffix, n_runs, comment)), "w")
    seed = 42
    wf.write(json.dumps(vars(args))+"\n")
    wf.flush()
    for i in range(n_runs):
        args.seed = seed + i
        print("seed", args.seed)
        metrics = main(args)
        m_str = "\t".join(["{:.4f}".format(x) for x in metrics])
        wf.write(m_str + "\n")
        wf.flush()
    wf.close()


def cal_avg_performance(n_runs=5, args=args, comment=None):
    now = datetime.now()
    now = now.strftime("%Y%m%d")
    out_dir = join(settings.RESULT_DIR, "self-ablation", now)
    os.makedirs(out_dir, exist_ok=True)
    cur_metrics = []
    with open(join(out_dir, "{}_robust_self_{}_lr_{}_kernelsize_{}_esb_{}_epsilon_{}_convtype_{}_convtune_{}_perfectdata_{}_suffix_{}_runs_{}_{}.txt".format(
        settings.data_source, args.loss_type, args.lr, args.mat1_kernel_size1, args.esb, args.epsilon, args.conv_type, args.conv_tune,
        args.perfect_data, args.suffix, n_runs, comment))) as rf:
        for i, line in enumerate(rf):
            if i == 0:
                continue
            items = [float(x) for x in line.strip().split()]
            cur_metrics.append(items)
    # m_sorted = sorted(cur_metrics, key=lambda x: x[-1], reverse=True)
    m_avg = np.mean(np.array(cur_metrics), axis=0)
    print("***************************************************")
    print("m avg", m_avg)
    print("***************************************************")


if __name__ == "__main__":
    print("args", args)
    # main(args=args)
    # multiple_robust_train(n_runs=5)
    # cal_avg_performance(n_runs=5)
    eval_per_person_nd_results(args=args)
    print()
    print("args", args)
    logger.info("done")
