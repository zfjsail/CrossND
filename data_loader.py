import gc
import sys
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

import utils

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp


class AuthorPaperPairMatchDataset(Dataset):
    def __init__(self, file_dir, matrix_size, n_papers_per_author, seed, shuffle, args, role="train",
                 perfect_data=False, suffix=None):
        self.file_dir = file_dir

        if role == "train":
            if str(suffix) == "None":

                if perfect_data:
                    mats_preprocess = utils.joblib_load_obj(file_dir, "paper_author_matching_input_mat_perfect.pkl")
                    stat_features = utils.joblib_load_obj(file_dir, "perfect_training_data_stat_features.pkl")
                    labels = utils.joblib_load_obj(file_dir, "training_pairs_perfect_labels.pkl")
                    assert len(mats_preprocess) == len(stat_features) == len(labels)
                    scaler = utils.joblib_load_obj(file_dir, "scaler.obj")
                    stat_features = scaler.transform(stat_features)
                else:
                    mats_preprocess = utils.joblib_load_obj(file_dir, "paper_author_matching_pairs_input_mat_mid.pkl")
                    stat_features = utils.joblib_load_obj(file_dir, "train_stat_features_mid.pkl")
                    labels = utils.joblib_load_obj(file_dir, "pa_labels_mid.pkl")
                    scaler = StandardScaler()
                    stat_features = scaler.fit_transform(stat_features)
                    utils.joblib_dump_obj(scaler, file_dir, "scaler.obj")
            else:
                mats_preprocess = utils.joblib_load_obj(file_dir, "paper_author_matching_pairs_input_mat_mid_{}.pkl".format(suffix))
                stat_features = utils.joblib_load_obj(file_dir, "train_stat_features_mid_{}.pkl".format(suffix))
                if perfect_data:
                    raise NotImplementedError
                else:
                    labels = utils.joblib_load_obj(file_dir, "pa_labels_mid_{}.pkl".format(suffix))

        elif role == "valid" or role == "test":
            mats_preprocess = utils.joblib_load_obj(file_dir,
                                                    "paper_author_matching_input_mat_eval_mid_{}.pkl".format(role))
            labels = utils.joblib_load_obj(file_dir, "{}_labels_mid.pkl".format(role))
            stat_features = utils.joblib_load_obj(file_dir, "{}_stat_features_mid.pkl".format(role))
            scaler = utils.joblib_load_obj(file_dir, "scaler.obj")
            stat_features = scaler.transform(stat_features)
        else:
            raise NotImplementedError

        N = len(labels)
        self.x = np.zeros(shape=(N, n_papers_per_author, matrix_size, matrix_size))
        self.Y = labels
        self.stat_features = np.array(stat_features)

        for i, cur_mat in enumerate(mats_preprocess):
            if i % 10000 == 0:
                logger.info("pair %d", i)
            if i >= args.debug_scale:
                break
            if len(cur_mat) == 0:
                continue
            n_cur_author_papers = len(cur_mat)
            for j in range(n_cur_author_papers):
                cur_sim_mat = cur_mat[j]
                s1 = min(cur_sim_mat.shape[0], matrix_size)
                s2 = min(cur_sim_mat.shape[1], matrix_size)
                self.x[i, j, : s1, : s2] = cur_sim_mat[: s1, :s2]

        if shuffle:
            logger.info("shuffling data...")
            self.x, self.Y, self.stat_features = sklearn.utils.shuffle(
                self.x, self.Y, self.stat_features, random_state=seed
            )
            logger.info("shuffling done")

        self.N = len(self.Y)

        n_classes = self.get_num_class()
        class_weight = self.N / (n_classes * np.bincount(self.Y))
        self.class_weight = torch.FloatTensor(class_weight)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.x[idx], self.Y[idx], self.stat_features[idx], idx

    def get_num_class(self):
        return np.unique(self.Y).shape[0]


class AuthorPaperTriIcsEvalDataset(Dataset):

    def __init__(self, file_dir, matrix_size, n_papers_per_author, args, role="valid", suffix=None):
        scaler = utils.joblib_load_obj(file_dir, "scaler.obj")
        if str(suffix) == "None":
            mats_preprocess = utils.joblib_load_obj(file_dir, "ics_input_mat_mid_{}.pkl".format(role))
            labels = utils.joblib_load_obj(file_dir, "ics_labels_mid_{}.pkl".format(role))
            stat_features = utils.joblib_load_obj(file_dir, "ics_stat_features_mid_{}.pkl".format(role))
            stat_features = scaler.transform(stat_features)
        else:
            mats_preprocess = utils.joblib_load_obj(file_dir, "ics_input_mat_mid_{}_{}.pkl".format(role, suffix))
            labels = utils.joblib_load_obj(file_dir, "ics_labels_mid_{}_{}.pkl".format(role, suffix))
            stat_features = utils.joblib_load_obj(file_dir, "ics_stat_features_mid_{}_{}.pkl".format(role, suffix))
            stat_features = scaler.transform(stat_features)

        N = len(labels)
        self.x = np.zeros(shape=(N, n_papers_per_author, matrix_size, matrix_size))
        self.Y = labels
        self.stat_features = np.array(stat_features)

        for i, cur_mat in enumerate(mats_preprocess):
            if i % 10000 == 0:
                logger.info("pair %d", i)
            if i >= args.debug_scale:
                break
            if len(cur_mat) == 0:
                continue
            n_cur_author_papers = len(cur_mat)
            for j in range(n_cur_author_papers):
                cur_sim_mat = cur_mat[j]
                s1 = min(cur_sim_mat.shape[0], matrix_size)
                s2 = min(cur_sim_mat.shape[1], matrix_size)
                self.x[i, j, : s1, : s2] = cur_sim_mat[: s1, :s2]

        self.N = len(self.Y)

        n_classes = self.get_num_class()
        class_weight = self.N / (n_classes * np.bincount(self.Y))
        self.class_weight = torch.FloatTensor(class_weight)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.x[idx], self.stat_features[idx], self.Y[idx]

    def get_num_class(self):
        return np.unique(self.Y).shape[0]


class AuthorPaperMatchFuzzyNegDataset(Dataset):

    def __init__(self, file_dir, matrix_size, n_papers_per_author, seed, shuffle, args):
        self.file_dir = file_dir
        self.matrix_size = matrix_size

        labels = utils.joblib_load_obj(file_dir, "fuzzy_neg_labels_mid_{}.pkl".format(args.ics_suffix))
        # labels = utils.joblib_load_obj(file_dir, "pa_labels_mid.pkl")
        scaler = utils.joblib_load_obj(file_dir, "scaler.obj")

        stat_features = utils.joblib_load_obj(file_dir, "fuzzy_neg_pairs_train_stat_features_mid_{}.pkl".format(args.ics_suffix))
        # stat_features = utils.joblib_load_obj(file_dir, "train_stat_features_mid.pkl")
        stat_features = scaler.transform(stat_features)

        mats_in = utils.joblib_load_obj(file_dir, "pa_fuzzy_neg_input_mat_in_mid_{}.pkl".format(args.ics_suffix))
        # mats_in = utils.joblib_load_obj(file_dir, "paper_author_matching_pairs_input_mat_mid.pkl")

        N = len(labels)
        self.x = np.zeros(shape=(N, n_papers_per_author, matrix_size, matrix_size), dtype=np.float32)
        self.Y = labels
        self.stat_features = stat_features

        for i, cur_mat in enumerate(mats_in):
            if i % 10000 == 0:
                logger.info("pair %d", i)
            if i >= args.debug_scale:
                break
            if len(cur_mat) == 0:
                continue
            n_cur_author_papers = len(cur_mat)
            for j in range(n_cur_author_papers):
                cur_sim_mat = cur_mat[j]
                s1 = min(cur_sim_mat.shape[0], matrix_size)
                s2 = min(cur_sim_mat.shape[1], matrix_size)
                self.x[i, j, : s1, : s2] = cur_sim_mat[: s1, :s2]

        del mats_in
        gc.collect()

        x_size = sys.getsizeof(self.x)/1e9
        print("x_size", x_size)

        if shuffle:
            logger.info("shuffling data...")
            self.x, self.Y, self.stat_features = sklearn.utils.shuffle(
                self.x, self.Y, self.stat_features, random_state=seed
            )

            # permutation = np.random.permutation(len(self.Y))
            # logger.info("permutation generated")
            # self.Y = np.array(self.Y)[permutation]
            # self.stat_features = np.array(self.stat_features)[permutation]
            # logger.info("simple features done. hard features begin...")
            # self.x = self.x[permutation]

            logger.info("shuffling done")

        self.N = len(self.Y)

        n_classes = self.get_num_class()
        class_weight = self.N / (n_classes * np.bincount(self.Y))
        self.class_weight = torch.FloatTensor(class_weight)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.x[idx], self.Y[idx], self.stat_features[idx], idx

    def get_num_class(self):
        return np.unique(self.Y).shape[0]


class AuthorPaperMatchCsIcsDataset(Dataset):

    def __init__(self, file_dir, matrix_size, n_papers_per_author, ics_suffix, seed, shuffle):
        self.file_dir = file_dir
        self.matrix_size = matrix_size

        # labels = utils.joblib_load_obj(file_dir, "cs_and_ics_train_labels_mid.pkl")
        # author_sim_scores = utils.joblib_load_obj(file_dir, "cs_and_ics_train_author_sim_scores_mid.pkl")
        # mats_in = utils.joblib_load_obj(file_dir, "cs_and_ics_train_input_mat_in_mid.pkl")
        # mats_out = utils.joblib_load_obj(file_dir, "cs_and_ics_train_input_mat_out_mid.pkl")

        labels = utils.joblib_load_obj(file_dir, "cs_and_ics_train_labels_mid_{}.pkl".format(ics_suffix))
        truths = utils.joblib_load_obj(file_dir, "cs_and_ics_train_truths_mid_{}.pkl".format(ics_suffix))
        author_sim_scores = utils.joblib_load_obj(file_dir, "cs_and_ics_train_author_sim_scores_mid_{}.pkl".format(ics_suffix))
        mats_in = utils.joblib_load_obj(file_dir, "cs_and_ics_train_input_mat_in_mid_{}.pkl".format(ics_suffix))
        mats_out = utils.joblib_load_obj(file_dir, "cs_and_ics_train_input_mat_out_mid_{}.pkl".format(ics_suffix))

        # self.features_add_in = utils.joblib_load_obj(file_dir, "cs_ics_triplets_train_stat_features_in_mid.pkl")
        # self.features_add_out = utils.joblib_load_obj(file_dir, "cs_ics_triplets_train_stat_features_out_mid.pkl")

        self.features_add_in = utils.joblib_load_obj(file_dir, "cs_ics_triplets_train_stat_features_in_mid_{}.pkl".format(ics_suffix))
        self.features_add_out = utils.joblib_load_obj(file_dir, "cs_ics_triplets_train_stat_features_out_mid_{}.pkl".format(ics_suffix))

        scaler = utils.joblib_load_obj(file_dir, "scaler.obj")
        self.features_add_in = scaler.transform(self.features_add_in)
        self.features_add_out = scaler.transform(self.features_add_out)

        assert len(self.features_add_in) == len(mats_in)

        N = len(labels)
        self.X1 = np.zeros(shape=(N, n_papers_per_author, matrix_size, matrix_size), dtype=np.float)
        self.X2 = np.zeros(shape=(N, n_papers_per_author, matrix_size, matrix_size), dtype=np.float)
        self.Y = labels
        self.truths = truths

        for i in range(len(labels)):
            if i % 10000 == 0:
                logger.info("pair %d", i)
            cur_mat1 = mats_in[i]
            if len(cur_mat1) == 0:
                continue
            n_cur_author_papers = len(cur_mat1)
            for j in range(n_cur_author_papers):
                cur_sim_mat = cur_mat1[j]
                self.X1[i, j, : cur_sim_mat.shape[0], : cur_sim_mat.shape[1]] = cur_sim_mat

            cur_mat2 = mats_out[i]
            if len(cur_mat2) == 0:
                continue
            n_cur_author_papers = len(cur_mat2)
            for j in range(n_cur_author_papers):
                cur_sim_mat = cur_mat2[j]
                self.X2[i, j, : cur_sim_mat.shape[0], : cur_sim_mat.shape[1]] = cur_sim_mat

        del mats_in, mats_out
        gc.collect()

        if shuffle:
            self.X1, self.X2, self.Y, author_sim_scores, self.features_add_in, self.features_add_out, self.truths = sklearn.utils.shuffle(
                self.X1, self.X2, self.Y, author_sim_scores, self.features_add_in, self.features_add_out, self.truths,
                random_state=seed
            )

        self.author_sim_scores = author_sim_scores

        self.N = len(self.Y)

        n_classes = self.get_num_class()
        class_weight = self.N / (n_classes * np.bincount(self.Y))
        self.class_weight = torch.FloatTensor(class_weight)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.X1[idx], self.X2[idx], self.Y[idx], self.author_sim_scores[idx], self.features_add_in[idx], \
               self.features_add_out[idx], self.truths[idx]

    def get_num_class(self):
        return np.unique(self.Y).shape[0]
