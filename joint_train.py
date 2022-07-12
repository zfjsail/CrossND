import argparse
import os
from os.path import join
import json
import time
from datetime import datetime
from copy import deepcopy
from collections import defaultdict as dd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from data_loader import AuthorPaperMatchFuzzyNegDataset, AuthorPaperMatchCsIcsDataset
from data_loader import AuthorPaperPairMatchDataset, AuthorPaperTriIcsEvalDataset
from model import CroNDBase
import utils
from utils import ChunkSampler
import settings

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--matrix-size1', type=int, default=settings.MAX_MAT_SIZE, help='Matrix size 1.')
parser.add_argument('--n-papers-per-author', type=int, default=10, help='Matrix size 1.')
parser.add_argument('--mat1-channel1', type=int, default=20, help='Matrix1 number of channels1.')
parser.add_argument('--mat1-hidden', type=int, default=64, help='Matrix1 hidden dim.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--forget-rate', type=float, default=0.1, help='Forget rate.')
parser.add_argument('--weight-decay', type=float, default=1e-2, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--total-epochs', type=int, default=100, help="Round Number")
parser.add_argument('--check-point', type=int, default=5, help="Check point")
parser.add_argument('--shuffle', action='store_true', default=True, help="Shuffle dataset")
# parser.add_argument('--file-dir', type=str, default=settings.OUT_DATASET_DIR, help="Input file directory")
parser.add_argument('--conv-type', type=str, default="kc", help="Conv type")
parser.add_argument('--conv-tune', type=str, default="yes", help="Tune conv paras")
parser.add_argument('--mat1-kernel-size1', type=int, default=4, help='Matrix1 kernel size1.')

parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--epsilon', type=float, default=0.6, help='Logit weight for soft labeling.')
parser.add_argument('--batch', type=int, default=4096, help="Batch size")
parser.add_argument('--epochs-old-model', type=int, default=1, help="Round Number")
parser.add_argument('--extra-loss-weight', type=int, default=3, help="Extra loss weight")
parser.add_argument('--tune-type', type=str, default="cmp", help="Cross-correction method")
parser.add_argument('--cmp-thr', type=float, default=0., help="CMP threshold")
parser.add_argument('--psl-thr', type=float, default=1., help="PSL threshold")
parser.add_argument('--loss-type', type=str, default="ls", help="Loss type of self-correction")
parser.add_argument('--ics-suffix', type=str, default="None", help="ICS suffix")
parser.add_argument('--self-cor', type=bool, default=True, help="Whether to perform self-correction")
parser.add_argument('--cross-cor', type=bool, default=True, help="Whether to perform cross-correction")
parser.add_argument('--debug', type=bool, default=False, help="Training ratio (0, 100)")
parser.add_argument('--debug-scale', type=int, default=10000, help="Training ratio (0, 100)")

args = parser.parse_args()

# post-process config -- not for tuning
if settings.data_source == "aminer":
    args.mat1_kernel_size1 = 4
    # args.conv_type = "cnn"
    # args.conv_tune = "no"
    args.extra_loss_weight = 3
    args.ics_suffix = "pctpospmin03negpmax009"
elif settings.data_source == "kddcup":
    args.mat1_kernel_size1 = 3
    args.extra_loss_weight = 1
    args.batch = 256
else:
    raise NotImplementedError


def forget_rate_scheduler(epochs, forget_rate, num_gradual, exponent):
    """Tells Co-Teaching what fraction of examples to forget at each epoch."""
    # define how many things to forget at each rate schedule
    forget_rate_schedule = np.ones(epochs) * forget_rate
    forget_rate_schedule[:num_gradual] = np.linspace(
        0, forget_rate ** exponent, num_gradual)
    return forget_rate_schedule


def loss_cross_entropy(epoch, y, t, ind, loss_all, class_weight):
    # Record loss and loss_div for further analysis
    loss = F.cross_entropy(y, t, class_weight, reduction="none")
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


def label_correction_loss(logits, label, class_weight, epsilon=args.epsilon):
    # sample_weights = class_weight[label]
    label = label.unsqueeze(1)
    label = torch.cat((1 - label, label), dim=1)
    new_target_probs = (1 - epsilon) * label + epsilon * logits
    num_examples = logits.shape[0]
    loss = torch.sum(new_target_probs * (-torch.log(logits + 1e-6)), 1)
    # loss = loss * sample_weights
    loss = sum(loss) / num_examples
    return loss


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


def cmp_checking_loss_from_init(out1, out2, labels, out1_init, out2_init, class_weight, truths):
    custom_thr = args.cmp_thr

    pos_probs = torch.cat((out1[:, 1].unsqueeze(1), out2[:, 1].unsqueeze(1)), dim=1)
    pos_probs_init = torch.cat((out1_init[:, 1].unsqueeze(1), out2_init[:, 1].unsqueeze(1)), dim=1)
    arg_min_idx = torch.argmin(pos_probs_init, dim=1)

    # arg_min_idx = torch.abs(truths)  # cheat

    pred_small_idx = (torch.exp(out1_init[:, 1]) >= custom_thr)
    arg_min_idx = torch.minimum(pred_small_idx, arg_min_idx)

    idx0 = range(0, len(labels))
    pos_probs_select = pos_probs[idx0, arg_min_idx]

    neg_probs = torch.cat((out1[:, 0].unsqueeze(1), out2[:, 0].unsqueeze(1)), dim=1)
    neg_probs_select = neg_probs[idx0, arg_min_idx]

    out_right = labels.float() * out1[:, 1] + (1 - labels.float()) * pos_probs_select
    out_left = labels.float() * out1[:, 0] + (1 - labels.float()) * neg_probs_select
    out = torch.cat((out_left.unsqueeze(1), out_right.unsqueeze(1)), dim=1)
    loss = F.nll_loss(out, labels, class_weight)
    return loss


def psl_loss(out1, out2, labels, out1_init, out2_init, author_sim, class_weight):
    m = args.psl_thr
    m_pos = m
    psl_thr = 1

    out1 = torch.exp(out1)
    out2 = torch.exp(out2)
    out1_init = torch.exp(out1_init)
    out2_init = torch.exp(out2_init)
    w1 = torch.sqrt(class_weight[1])
    w0 = torch.sqrt(class_weight[0])

    pos_probs = torch.cat((out1[:, 1].unsqueeze(1), out2[:, 1].unsqueeze(1)), dim=1)
    pos_probs_init = torch.cat((out1_init[:, 1].unsqueeze(1), out2_init[:, 1].unsqueeze(1)), dim=1)
    arg_min_idx = torch.argmin(pos_probs_init, dim=1)
    pred_small_idx = (out1_init[:, 1] >= psl_thr)
    pred_small_idx = torch.minimum(pred_small_idx, arg_min_idx)
    pred_big_idx = 1 - pred_small_idx

    idx0 = range(0, len(labels))
    larger_probs = pos_probs_init[idx0, pred_big_idx].detach().clone()
    small_probs = pos_probs[idx0, pred_small_idx]

    out = labels.float() * (author_sim.float() + out2_init[:, 1].detach().clone() - out1[:, 1] - m_pos) * w1 + \
          (1 - labels.float()) * (-author_sim.float() + torch.maximum(larger_probs, 1-larger_probs) - m + small_probs) * w0
    loss = (torch.min(torch.zeros_like(out), out) + m - 1)  # ** 2
    loss = loss.mean()
    return loss


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

    out_dir = join(settings.RESULT_DIR, "join-train", now)

    for i in range(0, 5):
        cur_seed = args.seed + i
        logger.info("cur_seed %s", cur_seed)
        model = CroNDBase(in_channels=args.n_papers_per_author, matrix_size=args.matrix_size1, channel1=args.mat1_channel1, 
                          kernel_size1=args.mat1_kernel_size1, hidden_size=args.mat1_hidden, conv_type=args.conv_type)
        model = model.float()
        model.load_state_dict(torch.load(join(model_dir, 'model_self_{}_tune_{}_cmp_{}_psl_{}_seed_{}.mdl'.format(
                              args.loss_type, args.tune_type, args.cmp_thr, args.psl_thr, cur_seed))))
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
        
        with open(join(out_dir, "results_per_name_self_{}_tune_{}_cmp_{}_psl_{}_seed_{}.json".format(args.loss_type, args.tune_type, args.cmp_thr, args.psl_thr, cur_seed)), "w") as wf:
            aids = sorted(aid_to_metrics.keys())
            for aid in aids:
                wf.write("{}\t{}\t{:.4f}\t{:.4f}\n".format(aid, aid_to_name[aid], aid_to_metrics[aid][0], aid_to_metrics[aid][1]))
                wf.flush()


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
            aid_to_label[cur_aid].append(1 - cur_pair["label"])
            aid_to_score[cur_aid].append(scores[i + j])
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


def dev_simple_on_ics(data_loader, model):
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

    auc = roc_auc_score(labels, scores)
    maps = average_precision_score(labels, scores)
    model.train()
    return auc, maps


def joint_train(epoch, train_loader1, train_loader2, model, model2, optimizer, optimizer2, forget_rate_schedule,
                class_weight_fuzzy_neg, class_weight_triplets, loss_all, loss_div_all, args=args, model_init=None):
    model.train()
    if args.loss_type == "ct":
        model2.train()

    loss = 0.
    total = 0.
    total_self = 0

    loss2 = 0.
    loss_2 = 0.

    if args.self_cor:
        v_list = np.zeros(len(loss_all))
        idx_each_class_noisy = [[] for i in range(2)]
        loss_mae = nn.L1Loss()

        for i_batch, batch in enumerate(train_loader1):
            X, Y, f_add, idx = batch
            if args.cuda:
                X = X.cuda()
                Y = Y.cuda()
                f_add = f_add.cuda()
            bs = Y.shape[0]

            optimizer.zero_grad(set_to_none=True)

            output, out_linear = model(X.float(), f_add.float())

            if args.loss_type == "ct":
                output2, out_linear2 = model2(X.float(), f_add.float())
            else:
                out_linear2 = None

            # TODO class_weight_fuzzy_neg
            if args.loss_type == "ce":
                loss_train = loss_cross_entropy(epoch, out_linear, Y, idx, loss_all, class_weight_fuzzy_neg)
            elif args.loss_type == "cores":
                loss_train, loss_v = loss_cores(epoch, out_linear, Y, idx, loss_all, loss_div_all)
                v_list[idx] = loss_v
                for i in range(bs):
                    if loss_v[i] == 0:
                        idx_each_class_noisy[Y[i]].append(idx[i])
            elif args.loss_type == "ls":  # label smoothing
                loss_train = label_correction_loss(torch.exp(output), Y, class_weight_fuzzy_neg)
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
            total_self += bs
            loss_train.backward()
            optimizer.step()

            if args.loss_type == "ct":
                loss2 += bs * loss_2.item()
                optimizer2.zero_grad(set_to_none=True)
                loss_2.backward()
                optimizer2.step()

    if args.cross_cor:
        for i_batch, batch in enumerate(train_loader2):
            X1, X2, Y, author_sim, f_add_in, f_add_out, truths = batch
            if args.cuda:
                X1 = X1.cuda()
                X2 = X2.cuda()
                Y = Y.cuda()
                truths = truths.cuda()
                author_sim = author_sim.cuda()
                f_add_in = f_add_in.cuda()
                f_add_out = f_add_out.cuda()

            bs = Y.shape[0]
            optimizer.zero_grad(set_to_none=True)

            output1, outs1_2 = model(X1.float(), f_add_in.float())
            output2, outs2_2 = model(X2.float(), f_add_out.float())

            output1_init, _ = model_init(X1.float(), f_add_in.float())
            output2_init, _ = model_init(X2.float(), f_add_out.float())

            if args.tune_type == "cmp":
                # loss_train = F.nll_loss(output1, Y, class_weight_triplets) * args.extra_loss_weight
                loss_train = cmp_checking_loss_from_init(output1, output2, Y, output1_init, output2_init,
                                                         class_weight_triplets, truths) * args.extra_loss_weight
            elif args.tune_type == "psl":
                loss_train = psl_loss(output1, output2, Y, output1_init, output2_init, author_sim,
                                      class_weight_triplets) * args.extra_loss_weight
            else:
                raise NotImplementedError

            loss += bs * loss_train.item()
            total += bs
            loss_train.backward()
            optimizer.step()

    logger.info("train loss epoch %d: %f, %f", epoch, loss / total, loss2 / total)


def main(args=args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    logger.info('cuda is available %s', args.cuda)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print("seed", args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    args.file_dir = settings.OUT_DATASET_DIR

    if args.debug:
        args.file_dir = join(args.file_dir, "debug-{}".format(args.debug_scale))
    else:
        args.debug_scale = 10000000

    if args.self_cor:
        dataset_fuzzy_neg = AuthorPaperMatchFuzzyNegDataset(args.file_dir, args.matrix_size1, args.n_papers_per_author,
                                                            args.seed, args.shuffle, args)
        class_weight_fuzzy_neg = dataset_fuzzy_neg.class_weight
        logger.info("class_weight fuzzy neg=%.2f:%.2f", class_weight_fuzzy_neg[0], class_weight_fuzzy_neg[1])
        N1 = len(dataset_fuzzy_neg)

        train_loader1 = DataLoader(dataset_fuzzy_neg, batch_size=args.batch, sampler=ChunkSampler(N1, 0),
                                   num_workers=4, pin_memory=True)
        loss_all = np.zeros((N1, args.total_epochs))
        loss_div_all = np.zeros((N1, args.total_epochs))
        if args.cuda:
            class_weight_fuzzy_neg = class_weight_fuzzy_neg.cuda()
    else:
        class_weight_fuzzy_neg = None
        train_loader1 = None
        loss_all, loss_div_all = None, None

    model = CroNDBase(in_channels=args.n_papers_per_author, matrix_size=args.matrix_size1, channel1=args.mat1_channel1,
                      kernel_size1=args.mat1_kernel_size1, hidden_size=args.mat1_hidden, conv_type=args.conv_type, conv_tune=args.conv_tune == "yes")
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

    dataset_triplets = AuthorPaperMatchCsIcsDataset(args.file_dir, args.matrix_size1, args.n_papers_per_author, args.ics_suffix,
                                                    args.seed, args.shuffle)

    N2 = len(dataset_triplets)
    train_loader2 = DataLoader(dataset_triplets, batch_size=args.batch, sampler=ChunkSampler(N2, 0),
                               num_workers=4, pin_memory=True)

    class_weight_triplets = dataset_triplets.class_weight
    logger.info("********************************************************")
    logger.info("class_weight=%.2f:%.2f", class_weight_triplets[0], class_weight_triplets[1])

    dataset = AuthorPaperPairMatchDataset(args.file_dir, args.matrix_size1, args.n_papers_per_author, args.seed,
                                          shuffle=False, role="valid", args=args)
    N = len(dataset)

    valid_loader = DataLoader(dataset, batch_size=args.batch, sampler=ChunkSampler(N, 0))

    dataset = AuthorPaperPairMatchDataset(args.file_dir, args.matrix_size1, args.n_papers_per_author, args.seed,
                                          shuffle=False, role="test", args=args)
    N = len(dataset)

    test_loader = DataLoader(dataset, batch_size=args.batch, sampler=ChunkSampler(N, 0))

    dataset = AuthorPaperTriIcsEvalDataset(args.file_dir, args.matrix_size1, args.n_papers_per_author, args,
                                           role="valid", suffix=args.ics_suffix)
    N = len(dataset)
    valid_loader_ics = DataLoader(dataset, batch_size=args.batch, sampler=ChunkSampler(N, 0))

    dataset = AuthorPaperTriIcsEvalDataset(args.file_dir, args.matrix_size1, args.n_papers_per_author, args,
                                           role="test", suffix=args.ics_suffix)
    N = len(dataset)
    test_loader_ics = DataLoader(dataset, batch_size=args.batch, sampler=ChunkSampler(N, 0))

    if args.cuda:
        model.cuda()
        class_weight_triplets = class_weight_triplets.cuda()

    optimizer1 = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    valid_pairs = utils.load_json(settings.DATASET_DIR, "eval_na_checking_triplets_relabel1_valid.json")
    test_pairs = utils.load_json(settings.DATASET_DIR, "eval_na_checking_triplets_relabel1_test.json")

    # valid_pairs = utils.load_json(settings.DATASET_DIR, "eval_na_checking_triplets_relabel1_moreics_valid.json")
    # test_pairs = utils.load_json(settings.DATASET_DIR, "eval_na_checking_triplets_relabel1_moreics_test.json")

    best_model = None
    best_round = -1
    auc_test_final = 0
    map_test_final = 0
    metric_max_valid = 0
    auc_max_valid = 0
    map_max_valid = 0

    auc_ics_best_valid = 0
    map_ics_best_valid = 0
    auc_ics_best_test = 0
    map_ics_best_test = 0

    t_total = time.time()
    logger.info("training...")
    model_init = deepcopy(model)
    model_init.eval()

    now = datetime.now()
    now = now.strftime("%Y%m%d")
    out_dir = join(settings.OUT_DIR, settings.data_source, now)
    os.makedirs(out_dir, exist_ok=True)

    wf = open(join(out_dir, "{}_self_{}_tune_{}_cmp_{}_psl_{}_seed_{}_suffix_{}_epoch_metrics.txt".format(
        settings.data_source, args.loss_type, args.tune_type, args.cmp_thr, args.psl_thr, args.seed, args.ics_suffix)), "w")

    forget_rate_schedule = forget_rate_scheduler(args.total_epochs, args.forget_rate, int(args.total_epochs * 3 / 10), 2)

    cur_auc_valid, cur_map_valid, loss_valid = dev(args, model, valid_loader, pairs=valid_pairs)
    cur_auc_test, cur_map_test, loss_test = dev(args, model, test_loader, pairs=test_pairs)

    # wf.write("{:.6f},{:.6f}\n".format(cur_auc_test, cur_map_test))
    # wf.flush()

    cur_auc_ics_valid, cur_map_ics_valid = dev_simple_on_ics(valid_loader_ics, model)
    cur_auc_ics_test, cur_map_ics_test = dev_simple_on_ics(test_loader_ics, model)

    wf.write("{:.6f},{:.6f}\n".format(cur_auc_ics_test, cur_map_ics_test))
    wf.flush()

    logger.info("***********************************************************")
    logger.info("round %d, valid auc %.4f, valid map %.4f", -1, cur_auc_valid, cur_map_valid)
    logger.info("round %d, test auc %.4f, test map %.4f", -1, cur_auc_test, cur_map_test)
    logger.info("round %d, valid auc ics %.4f, valid map ics %.4f", -1, cur_auc_ics_valid, cur_map_ics_valid)
    logger.info("round %d, test auc ics %.4f, test map ics %.4f", -1, cur_auc_ics_test, cur_map_ics_test)
    logger.info("current metric valid %.4f", cur_auc_valid + cur_map_valid)

    for i in range(args.total_epochs):
        joint_train(i, train_loader1, train_loader2, model, model2, optimizer1, optimizer2, forget_rate_schedule,
                    class_weight_fuzzy_neg, class_weight_triplets, loss_all, loss_div_all, args=args,
                    model_init=model_init)

        if i < args.epochs_old_model:
            model_init = deepcopy(model)
            model_init.eval()
        if (i + 1) % args.check_point != 0:
            continue

        cur_auc_valid, cur_map_valid, loss_valid = dev(args, model, valid_loader, pairs=valid_pairs)
        cur_auc_test, cur_map_test, loss_test = dev(args, model, test_loader, pairs=test_pairs)

        # wf.write("{:.6f},{:.6f}\n".format(cur_auc_test, cur_map_test))
        # wf.flush()

        cur_auc_ics_valid, cur_map_ics_valid = dev_simple_on_ics(valid_loader_ics, model)
        cur_auc_ics_test, cur_map_ics_test = dev_simple_on_ics(test_loader_ics, model)

        wf.write("{:.6f},{:.6f}\n".format(cur_auc_ics_test, cur_map_ics_test))
        wf.flush()

        logger.info("***********************************************************")
        logger.info("round %d, valid auc %.4f, valid map %.4f", -1, cur_auc_valid, cur_map_valid)
        logger.info("round %d, test auc %.4f, test map %.4f", -1, cur_auc_test, cur_map_test)
        logger.info("round %d, valid auc ics %.4f, valid map ics %.4f", -1, cur_auc_ics_valid, cur_map_ics_valid)
        logger.info("round %d, test auc ics %.4f, test map ics %.4f", -1, cur_auc_ics_test, cur_map_ics_test)
        logger.info("current metric valid %.4f", cur_auc_valid + cur_map_valid)

        if metric_max_valid < cur_auc_valid + cur_map_valid:
            best_round = i
            best_model = deepcopy(model)
            auc_test_final = cur_auc_test
            map_test_final = cur_map_test
            metric_max_valid = cur_auc_valid + cur_map_valid
            auc_max_valid = cur_auc_valid
            map_max_valid = cur_map_valid

            auc_ics_best_valid = cur_auc_ics_valid
            map_ics_best_valid = cur_map_ics_valid
            auc_ics_best_test = cur_auc_ics_test
            map_ics_best_test = cur_map_ics_test

            torch.save(best_model.state_dict(), join(settings.OUT_DATASET_DIR, "best_model_kc.mdl"))

        logger.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        logger.info("best metric valid %.4f", metric_max_valid)
        logger.info("best round %d, best auc valid now %.4f, best map valid now %.4f", best_round, auc_max_valid,
                    map_max_valid)
        logger.info("best... test auc %.4f, test map %.4f", auc_test_final, map_test_final)
        logger.info("br best auc ics valid %.4f, best map ics valid %.4f", auc_ics_best_valid, map_ics_best_valid)
        logger.info("br best auc ics test %.4f, best map ics test %.4f", auc_ics_best_test, map_ics_best_test)

    logger.info("optimization Finished!")
    logger.info("total time elapsed: {:.4f}s".format(time.time() - t_total))

    now = datetime.now()
    now = now.strftime("%Y%m%d")

    model_dir = join(args.file_dir, "models", now)
    os.makedirs(model_dir, exist_ok=True)
    torch.save(best_model.state_dict(), join(model_dir, 'model_self_{}_tune_{}_cmp_{}_psl_{}_seed_{}.mdl'.format(
        args.loss_type, args.tune_type, args.cmp_thr, args.psl_thr, args.seed)))
    logger.info('paper author matching model saved')

    auc_best_model_valid, map_best_model_valid, loss_valid = dev(args, best_model, valid_loader, pairs=valid_pairs)
    auc_best_model_test, map_best_model_test, loss_test = dev(args, best_model, test_loader, pairs=test_pairs)
    auc_ics_best_test, map_ics_best_test = dev_simple_on_ics(test_loader_ics, best_model)

    logger.info("best metric valid %.4f", metric_max_valid)
    logger.info("Finally, best round %d\n*****best auc valid now %.4f, best map valid now %.4f\n"
                "*****best auc test %.4f, best map test now %.4f",
                best_round, auc_best_model_valid, map_best_model_valid, auc_best_model_test, map_best_model_test)
    logger.info("*****best auc ics test %.4f, best map ics test %.4f", auc_ics_best_test, map_ics_best_test)

    wf.close()

    return auc_best_model_test, map_best_model_test, auc_ics_best_test, map_ics_best_test, \
           auc_best_model_valid, map_best_model_valid, best_round, auc_best_model_valid + map_best_model_valid


def multiple_joint_train(n_runs=5, args=args, comment=None):
    now = datetime.now()
    now = now.strftime("%Y%m%d")
    out_dir = join(settings.RESULT_DIR, "join-train", now)
    os.makedirs(out_dir, exist_ok=True)
    fname = join(out_dir, "{}_self_{}_tune_{}_cmp_{}_psl_{}_lr_{}_kernelsize_{}_conv_type_{}_convtune_{}_epsilon_{}_epocholdmodel_{}_exweight_{}_selfcor_{}_crosscor_{}_icssuffix_{}_runs_{}_{}.txt".format(
        settings.data_source, args.loss_type, args.tune_type, args.cmp_thr, args.psl_thr, args.lr, args.mat1_kernel_size1, args.conv_type, args.conv_tune, args.epsilon,
        args.epochs_old_model, args.extra_loss_weight, args.self_cor, args.cross_cor, args.ics_suffix, n_runs, comment
    ))
    n_runs_old = 0
    if os.path.isfile(fname):
        n_runs_old = max(len(open(fname).readlines()) - 1, 0)
        wf = open(fname, "a")
    else:
        wf = open(fname, "w")
        wf.write(json.dumps(vars(args)) + "\n")
        wf.flush()
    print("***********************************************")
    print("n_runs_old", n_runs_old)
    seed = 42
    for i in range(n_runs_old, n_runs):
        args.seed = seed + i
        print("args", args)
        print("***********************************************")
        print("seed", args.seed)
        metrics = main(args)
        m_str = "\t".join(["{:.4f}".format(x) for x in metrics])
        wf.write(m_str + "\n")
        wf.flush()
    wf.close()


def cal_avg_performance(n_runs=5, args=args, comment=None):
    now = datetime.now()
    now = now.strftime("%Y%m%d")
    out_dir = join(settings.RESULT_DIR, "join-train", now)
    os.makedirs(out_dir, exist_ok=True)
    cur_metrics = []
    with open(join(out_dir, "{}_self_{}_tune_{}_cmp_{}_psl_{}_lr_{}_kernelsize_{}_conv_type_{}_convtune_{}_epsilon_{}_epocholdmodel_{}_exweight_{}_selfcor_{}_crosscor_{}_icssuffix_{}_runs_{}_{}.txt".format(
        settings.data_source, args.loss_type, args.tune_type, args.cmp_thr, args.psl_thr, args.lr, args.mat1_kernel_size1, args.conv_type, args.conv_tune, args.epsilon,
        args.epochs_old_model, args.extra_loss_weight, args.self_cor, args.cross_cor, args.ics_suffix, n_runs, comment
    ))) as rf:
        for i, line in enumerate(rf):
            if i == 0:
                continue
            items = [float(x) for x in line.strip().split()]
            cur_metrics.append(items)
    m_avg = np.mean(np.array(cur_metrics), axis=0)
    print("***************************************************")
    print("m avg", m_avg)
    print("***************************************************")


def analyze_pred_err(args=args):
    model = CroNDBase(in_channels=args.n_papers_per_author, matrix_size=args.matrix_size1, channel1=args.mat1_channel1,
                      kernel_size1=args.mat1_kernel_size1, hidden_size=args.mat1_hidden, conv_type=args.conv_type)
    model = model.float()
    model.load_state_dict(torch.load(join(settings.OUT_DATASET_DIR, "best_model_kc.mdl")))
    model.eval()
    model.cuda()

    args.file_dir = settings.OUT_DATASET_DIR

    valid_pairs = utils.load_json(settings.DATASET_DIR, "eval_na_checking_triplets_relabel1_valid.json")
    test_pairs = utils.load_json(settings.DATASET_DIR, "eval_na_checking_triplets_relabel1_test.json")

    dataset = AuthorPaperPairMatchDataset(args.file_dir, args.matrix_size1, args.n_papers_per_author, args.seed,
                                          shuffle=False, role="valid", args=args)
    N = len(dataset)

    valid_loader = DataLoader(dataset, batch_size=args.batch, sampler=ChunkSampler(N, 0))

    dataset = AuthorPaperPairMatchDataset(args.file_dir, args.matrix_size1, args.n_papers_per_author, args.seed,
                                          shuffle=False, role="test", args=args)
    N = len(dataset)

    test_loader = DataLoader(dataset, batch_size=args.batch, sampler=ChunkSampler(N, 0))

    # cur_auc_valid, cur_map_valid, loss_valid = dev(args, model, valid_loader, pairs=valid_pairs)
    # cur_auc_test, cur_map_test, loss_test = dev(args, model, test_loader, pairs=test_pairs)

    cur_loader = valid_loader
    cur_pairs = valid_pairs

    labels = []
    scores = []
    scores_1 = []

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    logger.info('cuda is available %s', args.cuda)

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
            # cur_loss = F.nll_loss(batch_score, Y)

    for i in range(len(cur_pairs)):
        cur_pred = scores_1[i]
        cur_pairs[i]["pred"] = cur_pred

    cur_pairs_sorted = sorted(cur_pairs, key=lambda x: x["pred"], reverse=True)
    cur_pairs_neg_sorted = [x for x in cur_pairs_sorted if x["label"] == 0]
    print("n_neg", len(cur_pairs_neg_sorted))
    utils.dump_json(cur_pairs_neg_sorted, settings.OUT_DATASET_DIR, "err_pairs_pred_sorted.json")


if __name__ == "__main__":
    print("args", args)
    # main(args=args)
    # multiple_joint_train(n_runs=5)
    # cal_avg_performance(n_runs=5)
    # analyze_pred_err(args=args)
    eval_per_person_nd_results(args=args)
    print("args", args)
    logger.info("done")
