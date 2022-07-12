from os.path import join
import os
import json
import time
import math
import numpy as np
from copy import deepcopy
from bson import ObjectId
from fuzzywuzzy import fuzz
import random
from tqdm import tqdm
from collections import defaultdict as dd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score, average_precision_score

from address_normalization import addressNormalization
from get_paper_id import get_paper_id_from_title
from client import LMDBClient, MongoDBClientKexie
import utils
from utils import Singleton
import settings

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp

_emb_model = None


def get_pub_feature(x):
    i, pid, paper = x
    if "title" not in paper or "authors" not in paper:
        return None
    if len(paper["authors"]) > 1000:
        return None
    if len(paper["authors"]) > 30:
        print(i, pid, len(paper["authors"]))
    n_authors = len(paper.get('authors', []))
    authors = []
    for j in range(n_authors):
        author_features, word_features = utils.extract_author_features(paper, j)
        aid = '{}-{}'.format(pid, j)
        authors.append((aid, author_features, word_features))
    if i % 1000 == 0:
        print("features", authors)
    lcw = LMDBClient(name="pub_mid.features")
    lcw.set(pid, authors)
    lcw.close()


def dump_paper_id_to_raw_features():
    """
    generate author features by raw publication data and dump to files
    author features are defined by his/her paper attributes excluding the author's name
    """
    # Load publication features
    if settings.data_source == "aminer":
        _pubs_dict = utils.load_json(settings.DATASET_DIR, 'conna_pub_dict_mid.json')
    elif settings.data_source == "kddcup":
        _pubs_dict = utils.load_json(settings.DATASET_DIR, 'conna_pub_dict.json')
    else:
        raise NotImplementedError
    paper_list = [(i, pid, _pubs_dict[pid]) for i, pid in enumerate(_pubs_dict)]
    utils.processed_by_multi_thread(get_pub_feature, paper_list)


@Singleton
class EmbeddingModel:
    author_model = None
    word_model = None

    def train(self):
        lcw = LMDBClient(name="pub_mid.features")
        if settings.data_source == "aminer":
            pubs_dict = utils.load_json(settings.DATASET_DIR, 'conna_pub_dict_mid.json')
        elif settings.data_source == "kddcup":
            pubs_dict = utils.load_json(settings.DATASET_DIR, 'conna_pub_dict.json')
        else:
            raise NotImplementedError
        index = 0
        author_data = []
        word_data = []
        for pid in tqdm(pubs_dict):
            pub_features = lcw.get(pid)
            if pub_features is None:
                continue
            for author_index in range(len(pub_features)):
                aid, author_features, word_features = pub_features[author_index]

                if index % 100000 == 0:
                    print(index, author_features, word_features)
                index += 1

                random.shuffle(author_features)
                author_data.append(author_features)
                random.shuffle(word_features)
                word_data.append(word_features)

        self.author_model = Word2Vec(
            author_data, size=settings.EMB_DIM, window=5, min_count=5, workers=20,
        )
        out_dir = settings.EMB_DATA_DIR
        os.makedirs(out_dir, exist_ok=True)
        self.author_model.save(join(out_dir, 'author_name.emb'))
        self.word_model = Word2Vec(
            word_data, size=settings.EMB_DIM, window=5, min_count=5, workers=20,
        )
        self.word_model.save(join(out_dir, 'word.emb'))

    def load_author_name_emb(self):
        self.author_model = Word2Vec.load(join(settings.EMB_DATA_DIR, 'author_name.emb'))
        return self.author_model

    def load_word_name_emb(self):
        self.word_model = Word2Vec.load(join(settings.EMB_DATA_DIR, 'word.emb'))
        return self.word_model


def get_feature_index(i):
    word = _emb_model.wv.index2word[i]
    embedding = _emb_model.wv[word]
    return i, embedding


def dump_emb_array(emb_model, output_name):
    global _emb_model
    _emb_model = emb_model
    # transform the feature embeddings from embedding to (id, embedding)
    res = list(map(get_feature_index, range(len(_emb_model.wv.vocab))))
    sorted_embeddings = sorted(res, key=lambda x: x[0])
    word_embeddings = list(list(zip(*sorted_embeddings))[1])
    utils.dump_data(np.array(word_embeddings), settings.EMB_DATA_DIR, output_name)


def get_feature_ids_idfs_for_one_pub(features, emb_model, idfs):
    id_list = []
    idf_list = []
    for feature in features:
        if feature not in emb_model.wv:
            continue
        id = emb_model.wv.vocab[feature].index
        idf = 1
        if idfs and feature in idfs:
            idf = idfs[feature]
        id_list.append(id)
        idf_list.append(idf)
    return id_list, idf_list


def cal_feature_idf():
    """
    calculate word IDF (Inverse document frequency) using publication data
    """
    # features = utils.load_data(settings.EMB_DATA_DIR, "pub.features")
    lcw = LMDBClient(name="pub_mid.features")
    if settings.data_source == "aminer":
        pubs_dict = utils.load_json(settings.DATASET_DIR, 'conna_pub_dict_mid.json')
    elif settings.data_source == "kddcup":
        pubs_dict = utils.load_json(settings.DATASET_DIR, "conna_pub_dict.json")
    else:
        raise NotImplementedError

    feature_dir = join(settings.DATASET_DIR, 'global')
    os.makedirs(feature_dir, exist_ok=True)
    index = 0
    author_counter = dd(int)
    author_cnt = 0
    word_counter = dd(int)
    word_cnt = 0
    none_count = 0
    # for pub_index in range(len(features)):
    for pid in tqdm(pubs_dict):
        # pub_features = features[pub_index]
        pub_features = lcw.get(pid)
        # print(pub_features)
        if pub_features is None:
            none_count += 1
            continue
        for author_index in range(len(pub_features)):
            aid, author_features, word_features = pub_features[author_index]

            # if index % 100000 == 0:
            #     print(index, aid)
            index += 1

            for af in author_features:
                author_cnt += 1
                author_counter[af] += 1

            for wf in word_features:
                word_cnt += 1
                word_counter[wf] += 1

    author_idf = {}
    for k in author_counter:
        author_idf[k] = math.log(author_cnt / author_counter[k])

    word_idf = {}
    for k in word_counter:
        word_idf[k] = math.log(word_cnt / word_counter[k])

    utils.dump_data(dict(author_idf), settings.DATASET_DIR, "author_feature_idf.pkl")
    utils.dump_data(dict(word_idf), settings.DATASET_DIR, "word_feature_idf.pkl")
    print("None count: ", none_count)


def dump_paper_id_to_feature_token_ids():
    model = EmbeddingModel.Instance()
    word_emb_model = model.load_word_name_emb()
    word_emb_file = "word_emb.array"
    dump_emb_array(word_emb_model, word_emb_file)

    lcw = LMDBClient(name="pub_mid.features")
    if settings.data_source == "aminer":
        pubs_dict = utils.load_json(settings.DATASET_DIR, 'conna_pub_dict_mid.json')
    elif settings.data_source == "kddcup":
        pubs_dict = utils.load_json(settings.DATASET_DIR, 'conna_pub_dict.json')
    else:
        raise NotImplementedError
    word_idfs = utils.load_data(settings.DATASET_DIR, 'word_feature_idf.pkl')

    index = 0
    feature_dict = {}
    for pid in tqdm(pubs_dict):
        pub_features = lcw.get(pid)
        if pub_features is None:
            continue

        for author_index in range(len(pub_features)):
            aid, author_features, word_features = pub_features[author_index]
            if index % 100000 == 0:
                print(index, author_features, word_features)
            index += 1
            word_id_list, word_idf_list = get_feature_ids_idfs_for_one_pub(word_features, word_emb_model, word_idfs)

            if word_id_list is not None:
                feature_dict[aid] = (word_id_list, word_idf_list)
    utils.dump_data(feature_dict, settings.OUT_DATASET_DIR, "pub_feature.ids")


def gen_pa_pair_to_input_mat_dict():
    file_dir = settings.DATASET_DIR

    if settings.data_source == "aminer":
        name_aid_to_pids_in = utils.load_json(file_dir, "name_aid_to_pids_in_mid_filter.json")
        name_aid_to_pids_out = utils.load_json(file_dir, "name_aid_to_pids_out_mid_filter.json")
        pairs_valid = utils.load_json(file_dir, "eval_na_checking_triplets_relabel1_valid.json")
        pairs_test = utils.load_json(file_dir, "eval_na_checking_triplets_relabel1_test.json")

    elif settings.data_source == "kddcup":
        name_aid_to_pids_in = utils.load_json(file_dir, "train_author_pub_index_profile.json")
        # name_aid_to_pids_in = utils.load_json(file_dir, "train_author_pub_index_profile_enrich1.json")
        name_aid_to_pids_out = utils.load_json(file_dir, "aminer_name_aid_to_pids_with_idx.json")
        # name_aid_to_pids_out = utils.load_json(file_dir, "aminer_name_aid_to_pids_with_idx_enrich1.json")
        # pairs_valid = utils.load_json(file_dir, "eval_na_checking_triplets_relabel1_moreics_valid.json")
        # pairs_test = utils.load_json(file_dir, "eval_na_checking_triplets_relabel1_moreics_test.json")
        pairs_valid = utils.load_json(file_dir, "eval_na_checking_triplets_relabel1_valid.json")
        pairs_test = utils.load_json(file_dir, "eval_na_checking_triplets_relabel1_test.json")

    else:
        raise NotImplementedError

    name_pid_to_aid_out = dd(dict)
    for name in name_aid_to_pids_out:
        cur_name_dict = name_aid_to_pids_out[name]
        for aid in cur_name_dict:
            cur_pids = cur_name_dict[aid]
            for pid in cur_pids:
                name_pid_to_aid_out[name][pid] = aid

    triplets = []
    for i, name in enumerate(name_aid_to_pids_in):
        logger.info("name %d: %s", i, name)
        cur_name_dict = name_aid_to_pids_in[name]
        for aid in cur_name_dict:
            cur_pids = cur_name_dict[aid]
            for pid in cur_pids:
                if pid in name_pid_to_aid_out.get(name, {}):
                    aid_map = name_pid_to_aid_out[name][pid]
                    triplets.append({"aid1": aid, "aid2": aid_map, "pid": pid, "name": name})

    triplets_new = pairs_valid + pairs_test + triplets

    if settings.data_source == "aminer":
        pos_pairs_train = utils.load_json(file_dir, "positive_paper_author_pairs_conna_mid.json")
        neg_pairs_train = utils.load_json(file_dir, "negative_paper_author_pairs_conna_mid.json")
        paper_dict = utils.load_json(file_dir, "paper_dict_used_mag_mid.json")  # str: dict
    elif settings.data_source == "kddcup":
        pos_pairs_train = utils.load_json(file_dir, "positive_paper_author_pairs_conna_clean_1.json")
        # pos_pairs_train = utils.load_json(file_dir, "positive_paper_author_pairs_conna_more.json")
        neg_pairs_train = utils.load_json(file_dir, "negative_paper_author_pairs_conna_clean_1.json")
        # neg_pairs_train = utils.load_json(file_dir, "negative_paper_author_pairs_conna_more.json")
        paper_dict = utils.load_json(file_dir, "paper_dict_used_mag.json")  # str: dict
    else:
        raise NotImplementedError

    pair_to_mat_in = {}
    pair_to_mat_out = {}

    word_emb_mat = utils.load_data(join(file_dir, "emb"), "word_emb.array")
    pub_feature_dict = utils.load_data(settings.OUT_DATASET_DIR, "pub_feature.ids")

    valid_cnt1 = 0
    valid_cnt2 = 0

    pairs = pos_pairs_train + neg_pairs_train

    for i, pair in enumerate(triplets_new):
        pid = pair["pid"]
        aid1 = pair["aid1"]
        aid2 = pair["aid2"]
        name = pair["name"]

        cur_author_pubs = name_aid_to_pids_in[name][aid1]
        cur_pids = [x for x in cur_author_pubs if x != pid]
        cur_paper_year = paper_dict.get(str(pid.split("-")[0]), {}).get("year", 2022)  # this line works str(pid)

        if len(cur_pids) <= 10:
            pids_selected = cur_pids
        else:
            papers_attr = [(x, paper_dict[str(x.split("-")[0])]) for x in cur_pids if str(x.split("-")[0]) in paper_dict]
            papers_sorted = sorted(papers_attr, key=lambda x: abs(cur_paper_year - x[1].get("year", 2022)))
            pids_selected = [x[0] for x in papers_sorted][:10]

        cur_mats_in = []
        cur_mats_out = []
        flag1 = False
        flag2 = False
        cur_key_in = "{}~~~{}".format(aid1, pid)
        cur_key_out = "{}~~~{}".format(aid2, pid)

        if pid not in pub_feature_dict:
            pair_to_mat_in[cur_key_in] = cur_mats_in
            pair_to_mat_out[cur_key_out] = cur_mats_out
            continue

        word_id_list, word_idf_list = pub_feature_dict[pid]

        p_embs = word_emb_mat[word_id_list[: settings.MAX_MAT_SIZE]]

        if len(p_embs) == 0:
            print("p_emb 0", pid, word_id_list)
            pair_to_mat_in[cur_key_in] = cur_mats_in
            pair_to_mat_out[cur_key_out] = cur_mats_out
            continue

        for ap in pids_selected:
            if ap not in pub_feature_dict:
                continue
            ap_word_ids, _ = pub_feature_dict[ap]
            ap_embs = word_emb_mat[ap_word_ids[: settings.MAX_MAT_SIZE]]
            if len(ap_embs) > 0:
                cur_sim = cosine_similarity(p_embs, ap_embs)
                cur_mats_in.append(cur_sim)
                flag1 = True

        pair_to_mat_in[cur_key_in] = cur_mats_in
        if flag1:
            valid_cnt1 += 1
        else:
            print(pid, aid1, name, pids_selected)

        # out
        cur_author_pubs2 = name_aid_to_pids_out[name][str(aid2)]
        cur_pids2 = [x for x in cur_author_pubs2 if x != pid]
        if len(cur_pids2) <= 10:
            pids_selected2 = cur_pids2
        else:
            papers_attr2 = [(x, paper_dict[str(x.split("-")[0])]) for x in cur_pids2 if str(x.split("-")[0]) in paper_dict]
            papers_sorted2 = sorted(papers_attr2, key=lambda x: abs(cur_paper_year - x[1].get("year", 2022)))
            pids_selected2 = [x[0] for x in papers_sorted2][:10]

        for ap in pids_selected2:
            if ap not in pub_feature_dict:
                continue
            ap_word_ids, _ = pub_feature_dict[ap]
            ap_embs = word_emb_mat[ap_word_ids[: settings.MAX_MAT_SIZE]]
            if len(ap_embs) > 0:
                cur_sim = cosine_similarity(p_embs, ap_embs)
                cur_mats_out.append(cur_sim)
                flag2 = True

        pair_to_mat_out[cur_key_out] = cur_mats_out
        if flag2:
            valid_cnt2 += 1
        else:
            print(pid, aid2, name, pids_selected2)

        if i % 100 == 0:
            logger.info("pair %d, valid cnt1 %d, valid cnt2 %d", i, valid_cnt1, valid_cnt2)

        if i >= settings.TEST_SIZE - 1:
            break

    valid_cnt1 = 0
    valid_cnt2 = 0

    aid_to_name = {}
    for name in name_aid_to_pids_in:
        for aid in name_aid_to_pids_in[name]:
            aid_to_name[aid] = name

    for i, pair in enumerate(pairs):
        pid = pair["pid"]
        aid1 = pair["aid"]
        name = pair["name"]

        cur_key_in = "{}~~~{}".format(aid1, pid)
        if cur_key_in in pair_to_mat_in:
            valid_cnt1 += 1
            continue

        cur_author_pubs = name_aid_to_pids_in[aid_to_name[aid1]][aid1]
        cur_pids = [x for x in cur_author_pubs if x != pid]
        cur_paper_year = paper_dict.get(str(pid.split("-")[0]), {}).get("year", 2022)  # this line works str(pid)

        if len(cur_pids) <= 10:
            pids_selected = cur_pids
        else:
            papers_attr = [(x, paper_dict[str(x.split("-")[0])]) for x in cur_pids if str(x.split("-")[0]) in paper_dict]
            papers_sorted = sorted(papers_attr, key=lambda x: abs(cur_paper_year - x[1].get("year", 2022)))
            pids_selected = [x[0] for x in papers_sorted][:10]

        cur_mats_in = []
        # cur_mats_out = []
        flag1 = False
        flag2 = False

        if pid not in pub_feature_dict:
            pair_to_mat_in[cur_key_in] = cur_mats_in
            continue

        word_id_list, word_idf_list = pub_feature_dict[pid]
        p_embs = word_emb_mat[word_id_list[: settings.MAX_MAT_SIZE]]

        for ap in pids_selected:
            if ap not in pub_feature_dict:
                continue
            ap_word_ids, _ = pub_feature_dict[ap]
            ap_embs = word_emb_mat[ap_word_ids[: settings.MAX_MAT_SIZE]]
            if len(ap_embs) > 0:
                cur_sim = cosine_similarity(p_embs, ap_embs)
                cur_mats_in.append(cur_sim)
                flag1 = True

        pair_to_mat_in[cur_key_in] = cur_mats_in
        if flag1:
            valid_cnt1 += 1
        else:
            print(pid, aid1, name, pids_selected)

        if i % 100 == 0:
            logger.info("pair %d, valid cnt1 %d, valid cnt2 %d", i, valid_cnt1, valid_cnt2)

        if i >= settings.TEST_SIZE - 1:
            break

    print(len(pair_to_mat_in), len(pair_to_mat_out))
    utils.joblib_dump_obj(pair_to_mat_in, settings.OUT_DATASET_DIR, "pa_pair_to_mat_mid_in.pkl")
    utils.joblib_dump_obj(pair_to_mat_out, settings.OUT_DATASET_DIR, "pa_pair_to_mat_mid_out.pkl")


def gen_pa_pairs_to_input_mat_train():
    file_dir = settings.DATASET_DIR

    if settings.data_source == "aminer":
        pos_pairs = utils.load_json(file_dir, "positive_paper_author_pairs_conna_mid.json")
        neg_pairs = utils.load_json(file_dir, "negative_paper_author_pairs_conna_mid.json")
    elif settings.data_source == "kddcup":
        pos_pairs = utils.load_json(file_dir, "positive_paper_author_pairs_conna_clean_1.json")
        # pos_pairs = utils.load_json(file_dir, "positive_paper_author_pairs_conna_more.json")
        neg_pairs = utils.load_json(file_dir, "negative_paper_author_pairs_conna_clean_1.json")
        # neg_pairs = utils.load_json(file_dir, "negative_paper_author_pairs_conna_more.json")
    else:
        raise NotImplementedError

    pairs = pos_pairs + neg_pairs
    labels = [1] * len(pos_pairs) + [0] * len(neg_pairs)
    pair_to_mat_in = utils.joblib_load_obj(settings.OUT_DATASET_DIR, "pa_pair_to_mat_mid_in.pkl")

    attr_sim_mats = []
    empty_cnt = 0
    for pair in tqdm(pairs):
        pid = pair["pid"]
        aid = pair["aid"]
        cur_key_in = "{}~~~{}".format(aid, pid)
        if len(pair_to_mat_in[cur_key_in]) == 0:
            empty_cnt += 1
        attr_sim_mats.append(pair_to_mat_in[cur_key_in])
    print("empty cnt", empty_cnt)

    utils.joblib_dump_obj(attr_sim_mats, settings.OUT_DATASET_DIR, "paper_author_matching_pairs_input_mat_mid.pkl")
    utils.joblib_dump_obj(labels, settings.OUT_DATASET_DIR, "pa_labels_mid.pkl")


def gen_pa_pairs_eval_to_input_mat(role="valid"):
    file_dir = settings.DATASET_DIR

    if settings.data_source == "aminer":
        pairs = utils.load_json(file_dir, "eval_na_checking_triplets_relabel1_{}.json".format(role))
    # pairs = utils.load_json(file_dir, "eval_na_checking_triplets_relabel1_one2many_{}.json".format(role))
    elif settings.data_source == "kddcup":
        # pairs = utils.load_json(file_dir, "eval_na_checking_triplets_relabel1_moreics_{}.json".format(role))
        pairs = utils.load_json(file_dir, "eval_na_checking_triplets_relabel1_{}.json".format(role))
    else:
        raise NotImplementedError

    pair_to_mat_in = utils.joblib_load_obj(settings.OUT_DATASET_DIR, "pa_pair_to_mat_mid_in.pkl")
    pair_to_mat_out = utils.joblib_load_obj(settings.OUT_DATASET_DIR, "pa_pair_to_mat_mid_out.pkl")

    attr_sim_mats = []
    sim_mats_2 = []
    empty_cnt = 0
    empty_cnt_2 = 0
    for pair in tqdm(pairs):
        pid = pair["pid"]
        aid1 = pair["aid1"]
        aid2 = pair["aid2"]

        cur_key_in = "{}~~~{}".format(aid1, pid)
        cur_key_out = "{}~~~{}".format(aid2, pid)
        if len(pair_to_mat_in[cur_key_in]) == 0:
            empty_cnt += 1
        attr_sim_mats.append(pair_to_mat_in[cur_key_in])

        if len(pair_to_mat_out[cur_key_out]) == 0:
            empty_cnt_2 += 1
        sim_mats_2.append(pair_to_mat_out[cur_key_out])

    print("empty cnt", empty_cnt, "empty cnt 2", empty_cnt_2)

    assert len(attr_sim_mats) == len(sim_mats_2)

    print("number of mats", len(attr_sim_mats))
    utils.joblib_dump_obj(attr_sim_mats, settings.OUT_DATASET_DIR, "paper_author_matching_input_mat_eval_mid_{}.pkl".format(role))
    utils.joblib_dump_obj(sim_mats_2, settings.OUT_DATASET_DIR, "paper_author_matching_input_mat_eval_mid_out_{}.pkl".format(role))


def gen_ics_input_mat_mid(role="valid"):
    file_dir = settings.DATASET_DIR

    ics_triplets = utils.load_json(file_dir, "ics_triplets_relabel1_{}.json".format(role))
    pair_to_mat_in = utils.joblib_load_obj(settings.OUT_DATASET_DIR, "pa_pair_to_mat_mid_in.pkl")
    pair_to_mat_out = utils.joblib_load_obj(settings.OUT_DATASET_DIR, "pa_pair_to_mat_mid_out.pkl")

    attr_sim_mats = []
    sim_mats_2 = []

    for pair in tqdm(ics_triplets):
        pid = pair["pid"]
        aid = pair["aid1"]
        aid2 = pair["aid2"]

        cur_key_in = "{}~~~{}".format(aid, pid)
        cur_key_out = "{}~~~{}".format(aid2, pid)

        attr_sim_mats.append(pair_to_mat_in[cur_key_in])
        sim_mats_2.append(pair_to_mat_out[cur_key_out])

    print("number of mats", len(attr_sim_mats))
    utils.joblib_dump_obj(attr_sim_mats, settings.OUT_DATASET_DIR, "ics_input_mat_mid_{}.pkl".format(role))
    utils.joblib_dump_obj(sim_mats_2, settings.OUT_DATASET_DIR, "ics_input_mat_mid_out_{}.pkl".format(role))


def load_oag_linking_pairs(file_dir, fname, test_flag=False):
    aid_to_mid = {}
    with open(join(file_dir, fname)) as rf:
        for i, line in enumerate(rf):
            if i % 100000 == 0:
                logger.info("load linking pairs %d", i)
            cur_pair = json.loads(line)
            aid_to_mid[cur_pair["aid"]] = cur_pair["mid"]
            if test_flag and i > 10000000:
                break
    return aid_to_mid


def gen_cs_and_ics_triplets():
    file_dir = settings.DATASET_DIR
    aminer_name_aid_to_pids = utils.load_json(file_dir, "name_aid_to_pids_in_mid_filter.json")
    mag_name_aid_to_pids = utils.load_json(file_dir, "name_aid_to_pids_out_mid_filter.json")
    oag_linking_dir = join(settings.DATA_DIR, "..", "oag-2-1")
    aperson_to_mperson = load_oag_linking_pairs(oag_linking_dir, "author_linking_pairs_2020.txt")
    # paper_dict = utils.load_json(file_dir, "paper_dict_used_mag_mid.json")  # str: dict
    aminer_person_to_coauthors = utils.load_json(file_dir, "person_coauthors_in_mid.json")
    mag_person_to_coauthors = utils.load_json(file_dir, "person_coauthors_out_mid.json")

    mag_name_pid_to_aid = dd(dict)
    for name in mag_name_aid_to_pids:
        cur_name_dict = mag_name_aid_to_pids[name]
        for aid in cur_name_dict:
            cur_pids = cur_name_dict[aid]
            for pid in cur_pids:
                mag_name_pid_to_aid[name][pid] = aid

    pos_pairs = utils.load_json(file_dir, "positive_paper_author_pairs_conna_mid.json")
    aids_train = {item["aid"] for item in pos_pairs}

    pos_triplets = []
    neg_triplets = []
    n_paper_neg = 0
    n_venue_neg = 0
    n_coauthor_neg = 0
    n_map_by_others = 0

    n_align_pos = 0
    n_paper_pos = 0
    n_coauthor_pos = 0

    pubs_overlap_thr = 0.4
    venues_overlap_thr = 0.05
    # coauthors_overlap_thr = 0.05
    coauthors_overlap_thr = 0.0
    coauthors_overlap_thr_pos = 0.5

    pos_thr_pubs = 0.8
    pos_thr_coauthor = 1.0

    for i, name in enumerate(aminer_name_aid_to_pids):
        logger.info("name %d: %s", i, name)
        cur_name_dict = aminer_name_aid_to_pids[name]
        for aid in cur_name_dict:
            if aid not in aids_train:
                continue
            cur_pids = cur_name_dict[aid]
            # cur_pids_raw = [int(x.split("-")[0]) for x in cur_pids]

            # vids_1 = [paper_dict[str(x)].get("venue_id") for x in cur_pids_raw if
            #           str(x) in paper_dict and "venue_id" in paper_dict[str(x)]]

            coauthors_1 = aminer_person_to_coauthors.get(aid, [])

            for pid in cur_pids:
                if pid in mag_name_pid_to_aid.get(name, {}):
                    aid_map = mag_name_pid_to_aid[name][pid]
                    pubs_mag_author = mag_name_aid_to_pids.get(name, {}).get(str(aid_map), [])

                    if len(pubs_mag_author) < 5:
                        continue

                    mag_person_pids = pubs_mag_author
                    common_pids = set(cur_pids) & set(mag_person_pids)
                    n_common_pubs = len(common_pids)
                    pubs_overlap_a = n_common_pubs / len(cur_pids)
                    pubs_overlap_m = n_common_pubs / len(mag_person_pids)

                    # pubs_mag_author = [x.split("-")[0] for x in pubs_mag_author]
                    # vids_2 = [paper_dict[str(x)].get("venue_id") for x in pubs_mag_author if
                    #           str(x) in paper_dict and "venue_id" in paper_dict[str(x)]]

                    coauthors_2 = mag_person_to_coauthors.get(str(aid_map), [])
                    coauthor_sim, c_sim1, c_sim2 = utils.top_coauthor_sim(coauthors_1, coauthors_2, topk=10)

                    # v_sim = utils.top_venue_sim(vids_1, vids_2, topk=None)

                    if aid in aperson_to_mperson and int(aperson_to_mperson[aid]) == int(aid_map):
                        # pos_triplets.append(
                        #     {"aid1": aid, "aid2": int(aid_map), "pid": pid, "name": name, "author_sim": pubs_overlap_a})
                        # n_align_pos += 1
                        continue
                    elif min(pubs_overlap_a, pubs_overlap_m) >= pos_thr_pubs:
                        pos_triplets.append(
                            {"aid1": aid, "aid2": int(aid_map), "pid": pid, "name": name, "author_sim": pubs_overlap_a})
                        n_paper_pos += 1
                        continue
                    elif min(c_sim1, c_sim2) >= pos_thr_coauthor:
                        pos_triplets.append(
                            {"aid1": aid, "aid2": int(aid_map), "pid": pid, "name": name, "author_sim": coauthor_sim})
                        n_coauthor_pos += 1

                    if False:
                        pass
                    elif pubs_overlap_a < pubs_overlap_thr and pubs_overlap_m < pubs_overlap_thr:
                        neg_triplets.append({"aid1": aid, "aid2": int(aid_map), "pid": pid, "name": name,
                                             "author_sim": min(pubs_overlap_a, pubs_overlap_m)})
                        n_paper_neg += 1
                    # elif v_sim < venues_overlap_thr: # too few
                    #     neg_triplets.append({"aid1": aid, "aid2": int(aid_map), "pid": pid, "name": name,
                    #                          "author_sim": v_sim})
                    #     n_venue_neg += 1
                    elif coauthor_sim < coauthors_overlap_thr:
                        neg_triplets.append({"aid1": aid, "aid2": int(aid_map), "pid": pid, "name": name,
                                             "author_sim": coauthor_sim})
                        n_coauthor_neg += 1

    logger.info("n_align_pos %d, n_paper_pos %d, n_coauthor_pos %d", n_align_pos, n_paper_pos, n_coauthor_pos)
    logger.info("n_paper_neg %d, n_venue_neg %d, n_coauthors_neg %d, n_map_by_others %d",
                n_paper_neg, n_venue_neg, n_coauthor_neg, n_map_by_others)
    logger.info("number of postive triplets: %d", len(pos_triplets))
    logger.info("number of negative triplets: %d", len(neg_triplets))

    suffix_coa = "coa" + str(coauthors_overlap_thr)[0] + str(coauthors_overlap_thr)[2:]
    # print("suffix", suffix_coa)
    suffix_p = "p" + str(pubs_overlap_thr)[0] + str(pubs_overlap_thr)[2:]
    suffix_pos_p = "pp" + str(pos_thr_pubs)[0] + str(pos_thr_pubs)[2:]
    suffix_pos_coa = "pcoa" + str(pos_thr_coauthor)[0] + str(pos_thr_coauthor)[2:]
    suffix = suffix_coa + suffix_p + suffix_pos_coa + suffix_pos_p

    # utils.dump_json(pos_triplets, file_dir, "cs_triplets_via_author_sim.json")
    # utils.dump_json(neg_triplets, file_dir, "ics_triplets_via_author_sim.json")

    utils.dump_json(pos_triplets, file_dir, "cs_triplets_via_author_sim_{}.json".format(suffix))
    utils.dump_json(neg_triplets, file_dir, "ics_triplets_via_author_sim_{}.json".format(suffix))

    check_triplets_quality(suffix)


def check_triplets_quality(suffix):
    file_dir = settings.DATASET_DIR

    valid_triplets = utils.load_json(file_dir, "eval_na_checking_triplets_relabel1_valid.json")
    # valid_triplets = utils.load_json(file_dir, "eval_na_checking_triplets_relabel1_test.json")

    triplets_eval = valid_triplets

    # cs_triplets = utils.load_json(file_dir, "cs_triplets_via_author_sim.json")
    # ics_triplets = utils.load_json(file_dir, "ics_triplets_via_author_sim.json")

    file_dir = join(settings.DATASET_DIR, "cs_ics")

    cs_triplets = utils.load_json(file_dir, "cs_triplets_via_author_sim_{}.json".format(suffix))
    ics_triplets = utils.load_json(file_dir, "ics_triplets_via_author_sim_{}.json".format(suffix))

    pos_keys = set()
    neg_keys = set()

    for item in cs_triplets:
        aid1 = item["aid1"]
        pid = item["pid"]
        cur_key = aid1 + "~~~" + str(pid)
        pos_keys.add(cur_key)

    for item in ics_triplets:
        aid1 = item["aid1"]
        pid = item["pid"]
        cur_key = aid1 + "~~~" + str(pid)
        neg_keys.add(cur_key)

    y_pred = []
    y_true = []
    n_pred_0_true_1 = 0
    n_pred_1_true_0 = 0
    tt = 0
    ff = 0
    pred_true_cnt = 0
    real_true_cnt = 0
    pos_label_cnt = 0

    for item in triplets_eval:
        aid1 = item["aid1"]
        aid2 = item["aid2"]
        pid = item["pid"]
        cur_key = aid1 + "~~~" + str(pid)
        cur_label = item["label"]
        if cur_label == 1:
            pos_label_cnt += 1
        if cur_key in pos_keys:
            pred_true_cnt += 1
            y_pred.append(0)
            y_true.append(1 - cur_label)
            if cur_label == 0:
                n_pred_1_true_0 += 1
            else:
                tt += 1
                real_true_cnt += 1
        elif cur_key in neg_keys:
            y_pred.append(1)
            y_true.append(1 - cur_label)
            if cur_label == 1:
                n_pred_0_true_1 += 1
            else:
                ff += 1
        else:
            y_pred.append(0.5)
            y_true.append(1-cur_label)

    print(len(y_true), n_pred_0_true_1, n_pred_1_true_0, tt, ff)
    print("correct probs. real_true_cnt/pred_true_cnt", real_true_cnt / pred_true_cnt, real_true_cnt)
    print("positive label ratio", pos_label_cnt / len(triplets_eval), pos_label_cnt)
    auc = roc_auc_score(y_true, y_pred)
    maps = average_precision_score(y_true, y_pred)
    print("prec on neg_pred", ff/(ff+n_pred_0_true_1), ff)
    print("valid neg ratio", (ff + n_pred_1_true_0)/(pred_true_cnt + ff + n_pred_0_true_1))
    print("auc", auc, "maps", maps)


def gen_fuzzy_and_neg_triplets(suffixes=[]):
    file_dir = settings.DATASET_DIR

    # cs_triplets = utils.load_json(file_dir, "cs_triplets_via_author_sim.json")
    # ics_triplets = utils.load_json(file_dir, "ics_triplets_via_author_sim.json")

    if settings.data_source == "aminer":
        pos_pairs = utils.load_json(file_dir, "positive_paper_author_pairs_conna_mid.json")
        neg_pairs = utils.load_json(file_dir, "negative_paper_author_pairs_conna_mid.json")
        name_aid_to_pids_out = utils.load_json(file_dir, "name_aid_to_pids_out_mid_filter.json")
    elif settings.data_source == "kddcup":
        pos_pairs = utils.load_json(file_dir, "positive_paper_author_pairs_conna_more.json")
        neg_pairs = utils.load_json(file_dir, "negative_paper_author_pairs_conna_more.json")
        name_aid_to_pids_out = utils.load_json(file_dir, "aminer_name_aid_to_pids_with_idx_enrich1.json")
    else:
        raise NotImplementedError

    name_pid_to_aid_out = dd(dict)
    for name in name_aid_to_pids_out:
        cur_name_dict = name_aid_to_pids_out[name]
        for aid in cur_name_dict:
            cur_pids = cur_name_dict[aid]
            for pid in cur_pids:
                name_pid_to_aid_out[name][pid] = aid

    for suffix in suffixes:
        cs_triplets = utils.load_json(join(file_dir, "cs_ics"), "cs_triplets_via_author_sim_{}.json".format(suffix))
        ics_triplets = utils.load_json(join(file_dir, "cs_ics"), "ics_triplets_via_author_sim_{}.json".format(suffix))

        used_keys = set()

        for item in cs_triplets:
            aid1 = item["aid1"]
            pid = item["pid"]
            cur_key = aid1 + "~~~" + str(pid)
            if cur_key not in used_keys:
                used_keys.add(cur_key)
                # pass

        for item in ics_triplets:
            aid1 = item["aid1"]
            pid = item["pid"]
            cur_key = aid1 + "~~~" + str(pid)
            if cur_key not in used_keys:
                used_keys.add(cur_key)

        pos_triplets = []
        neg_triplets = []

        # name_pid_to_aid_in = dd(dict)

        for pair in pos_pairs:
            aid = pair["aid"]
            pid = pair["pid"]
            name = pair["name"]
            cur_key = aid + "~~~" + str(pid)
            # name_pid_to_aid_in[name][pid] = aid
            if cur_key not in used_keys:
                used_keys.add(cur_key)
                aid2 = aid
                pos_triplets.append({"aid1": aid, "aid2": aid2, "pid": pid, "name": name, "author_sim": 1})

        for pair in neg_pairs:
            aid = pair["aid"]
            pid = pair["pid"]
            name = pair["name"]
            cur_key = aid + "~~~" + str(pid)
            if cur_key not in used_keys:
                used_keys.add(cur_key)
                # aid2 = name_pid_to_aid_in[name][pid]
                # coauthors_1 = person_to_coauthors_out.get(aid, [])
                # coauthors_2 = person_to_coauthors_out.get(aid2, [])
                # coauthor_sim, _, _ = utils.top_coauthor_sim(coauthors_1, coauthors_2)
                aid2 = aid
                neg_triplets.append({"aid1": aid, "aid2": aid2, "pid": pid, "name": name, "author_sim": 1})

        logger.info("fuzzy pos %d, neg %d", len(pos_triplets), len(neg_triplets))
        utils.dump_json(pos_triplets, file_dir, "fuzzy_positive_remain_triplets_{}.json".format(suffix))
        utils.dump_json(neg_triplets, file_dir, "negative_remain_triplets.json")


def gen_fuzzy_neg_pa_pairs_input_mat(suffixes=[]):
    file_dir = settings.OUT_DATASET_DIR

    pair_to_mat_in = utils.joblib_load_obj(file_dir, "pa_pair_to_mat_mid_in.pkl")
    # pair_to_mat_out = utils.joblib_load_obj(file_dir, "pa_pair_to_mat_mid_out.pkl")

    for suffix in suffixes:
        pos_pairs = utils.load_json(settings.DATASET_DIR, "fuzzy_positive_remain_triplets_{}.json".format(suffix))
        neg_pairs = utils.load_json(settings.DATASET_DIR, "negative_remain_triplets.json")

        pairs = list(pos_pairs) + neg_pairs
        logger.info("number of pairs %d", len(pairs))
        labels = [1] * len(pos_pairs) + [0] * len(neg_pairs)
        # author_sim_scores = [x["author_sim"] for x in pos_pairs] + [x["author_sim"] for x in neg_pairs]

        attr_sim_mats = []
        # sim_mats_2 = []
        empty_cnt = 0
        empty_cnt_2 = 0
        for pair in tqdm(pairs):
            pid = pair["pid"]
            aid1 = pair["aid1"]
            aid2 = pair["aid2"]

            cur_key_in = "{}~~~{}".format(aid1, pid)
            # cur_key_out = "{}~~~{}".format(aid2, pid)
            if len(pair_to_mat_in[cur_key_in]) == 0:
                empty_cnt += 1
            attr_sim_mats.append(pair_to_mat_in[cur_key_in])

            # if len(pair_to_mat_out.get(cur_key_out, [])) == 0:
            #     empty_cnt_2 += 1
            # sim_mats_2.append(pair_to_mat_out.get(cur_key_out, []))

        print("empty cnt", empty_cnt, "empty cnt 2", empty_cnt_2)

        assert len(attr_sim_mats) == len(labels)

        utils.joblib_dump_obj(attr_sim_mats, file_dir, "pa_fuzzy_neg_input_mat_in_mid_{}.pkl".format(suffix))
        # utils.joblib_dump_obj(sim_mats_2, file_dir, "pa_fuzzy_neg_input_mat_out_mid.pkl")
        utils.joblib_dump_obj(labels, file_dir, "fuzzy_neg_labels_mid_{}.pkl".format(suffix))
        # utils.joblib_dump_obj(author_sim_scores, file_dir, "fuzzy_neg_triplets_author_sim_scores_mid.pkl")


def gen_cs_and_ics_triplets_input_mat(suffix=[]):
    file_dir = settings.OUT_DATASET_DIR

    if len(suffix) == 0:
        exit()

    pair_to_mat_in = utils.joblib_load_obj(file_dir, "pa_pair_to_mat_mid_in.pkl")
    pair_to_mat_out = utils.joblib_load_obj(file_dir, "pa_pair_to_mat_mid_out.pkl")

    for s in suffix:
        logger.info("current suffix %s", s)

    # pos_pairs = utils.load_json(settings.DATASET_DIR, "cs_triplets_via_author_sim.json")
    # neg_pairs = utils.load_json(settings.DATASET_DIR, "ics_triplets_via_author_sim.json")

        pos_pairs = utils.load_json(join(settings.DATASET_DIR, "cs_ics"), "cs_triplets_via_author_sim_{}.json".format(s))
        neg_pairs = utils.load_json(join(settings.DATASET_DIR, "cs_ics"), "ics_triplets_via_author_sim_{}.json".format(s))

        neg_pairs_copy = deepcopy(neg_pairs)
        dup_times = len(pos_pairs) // len(neg_pairs_copy)
        logger.info("dup times %d", dup_times)
        for _ in range(1, dup_times):
            neg_pairs += neg_pairs_copy

        logger.info("n_pos pairs %d, n_neg pairs %d", len(pos_pairs), len(neg_pairs))

        pos_pairs_sample = pos_pairs
        pairs = list(pos_pairs_sample) + neg_pairs
        logger.info("number of pairs %d", len(pairs))
        labels = [1] * len(pos_pairs_sample) + [0] * len(neg_pairs)
        author_sim_scores = [x["author_sim"] for x in pos_pairs_sample] + [x["author_sim"] for x in neg_pairs]
        truths = [x["label"] for x in pos_pairs_sample] + [x["label"] for x in neg_pairs]

        mats_in = []
        mats_out = []

        for pair in tqdm(pairs):
            pid = pair["pid"]
            aid1 = pair["aid1"]
            aid2 = pair["aid2"]

            cur_key_in = "{}~~~{}".format(aid1, pid)
            cur_key_out = "{}~~~{}".format(aid2, pid)

            mats_in.append(pair_to_mat_in[cur_key_in])
            mats_out.append(pair_to_mat_out[cur_key_out])

        labels = labels[: settings.TEST_SIZE]
        assert len(mats_in) == len(mats_out) == len(labels)
    # utils.joblib_dump_obj(mats_in, file_dir, "cs_and_ics_train_input_mat_in_mid.pkl")
    # utils.joblib_dump_obj(mats_out, file_dir, "cs_and_ics_train_input_mat_out_mid.pkl")
    # utils.joblib_dump_obj(labels, file_dir, "cs_and_ics_train_labels_mid.pkl")
    # utils.joblib_dump_obj(author_sim_scores, file_dir, "cs_and_ics_train_author_sim_scores_mid.pkl")

        utils.joblib_dump_obj(mats_in, file_dir, "cs_and_ics_train_input_mat_in_mid_{}.pkl".format(s))
        utils.joblib_dump_obj(mats_out, file_dir, "cs_and_ics_train_input_mat_out_mid_{}.pkl".format(s))
        utils.joblib_dump_obj(labels, file_dir, "cs_and_ics_train_labels_mid_{}.pkl".format(s))
        utils.joblib_dump_obj(author_sim_scores, file_dir, "cs_and_ics_train_author_sim_scores_mid_{}.pkl".format(s))
        utils.joblib_dump_obj(truths, file_dir, "cs_and_ics_train_truths_mid_{}.pkl".format(s))


def gen_debug_data():
    in_dir = settings.OUT_DATASET_DIR
    debug_scale = 10000
    debug_dir = join(settings.OUT_DATASET_DIR, "debug-{}".format(debug_scale))
    os.makedirs(debug_dir, exist_ok=True)
    half = int(debug_scale/2)
    files = [
        "fuzzy_neg_labels_mid.pkl",
        "fuzzy_neg_pairs_train_stat_features_mid.pkl",
        "pa_fuzzy_neg_input_mat_in_mid.pkl",
        "cs_and_ics_train_labels_mid.pkl",
        "cs_and_ics_train_author_sim_scores_mid.pkl",
        "cs_and_ics_train_input_mat_in_mid.pkl",
        "cs_and_ics_train_input_mat_out_mid.pkl",
        "cs_ics_triplets_train_stat_features_in_mid.pkl",
        "cs_ics_triplets_train_stat_features_out_mid.pkl",
        "paper_author_matching_pairs_input_mat_mid.pkl",
        "train_stat_features_mid.pkl",
        "pa_labels_mid.pkl",
    ]
    for f in tqdm(files):
        d = utils.joblib_load_obj(in_dir, f)
        part = d[:half] + d[-half:]
        print("part", len(part))
        utils.joblib_dump_obj(part, debug_dir, f)


def gen_cs_and_ics_percentage():
    # pos_percents = [0.1, 0.15, 0.2, 0.25, 0.3]
    # neg_percents = [0.01, 0.03, 0.05, 0.07, 0.09]

    pos_percents = [0.1, 0.2, 0.3, 0.4, 0.5]
    neg_percents = [0.12] * 5

    # pos_percents = [0.35, 0.4, 0.45, 0.5, 0.55]
    # neg_percents = [0.11, 0.13, 0.15, 0.17, 0.19]

    # triplets = utils.load_json(settings.OUT_DATASET_DIR, "triplets_train_author_sim_paper.json")
    if settings.data_source == "kddcup":
        triplets = utils.load_json(settings.OUT_DATASET_DIR, "triplets_train_author_sim_paper_more.json")
    elif settings.data_source == "aminer":
        triplets = utils.load_json(settings.OUT_DATASET_DIR, "triplets_train_author_sim_paper.json")
    else:
        raise NotImplementedError

    file_dir = join(settings.DATASET_DIR, "cs_ics")
    os.makedirs(file_dir, exist_ok=True)
    other_thr = 0.1
    ics_triplets_others = utils.load_json(file_dir, "ics_triplets_via_author_sim_extra_others_{}.json".format(other_thr))

    n_triplets_all = len(triplets)
    triplets_sorted = sorted(triplets, key=lambda x: x["author_sim"])
    triplets_sorted_max = sorted(triplets, key=lambda x: x["author_sim_max"])

    valid_triplets = utils.load_json(settings.DATASET_DIR, "eval_na_checking_triplets_relabel1_valid.json")
    # valid_triplets = utils.load_json(settings.DATASET_DIR, "eval_na_checking_triplets_relabel1_moreics_valid.json")

    # test_triplets = utils.load_json(settings.DATASET_DIR, "eval_na_checking_triplets_relabel1_test.json")
    aids_eval = {x["aid1"] for x in valid_triplets}
    pid_err_set = {x["pid"] for x in valid_triplets if x["label"] == 0}
    pid_true_set = {x["pid"] for x in valid_triplets if x["label"] == 1}
    err_rate_in_eval = len(pid_err_set)/(len(pid_err_set) + len(pid_true_set))
    print("err_rate_in_eval", err_rate_in_eval)

    for i in range(len(pos_percents)):
        pos_cnt = int(n_triplets_all * pos_percents[i])
        neg_cnt = int(n_triplets_all * neg_percents[i])
        cur_cs_triplets = triplets_sorted[-pos_cnt:]
        if settings.data_source == "aminer":
            cur_ics_triplets = triplets_sorted_max[:neg_cnt]
            print("ics thr", triplets_sorted_max[neg_cnt]["author_sim_max"])
            cur_suffix = "pospmin" + str(pos_percents[i])[0] + str(pos_percents[i])[2:] + "negpmax" + str(neg_percents[i])[0] + str(neg_percents[i])[2:] + "other" + str(other_thr)
        elif settings.data_source == "kddcup":
            cur_ics_triplets = triplets_sorted[:neg_cnt]
            print("ics thr", triplets_sorted[neg_cnt]["author_sim"])
            cur_suffix = "pctpospmin" + str(pos_percents[i])[0] + str(pos_percents[i])[2:] + "negpmin" + str(neg_percents[i])[0] + str(neg_percents[i])[2:]
        else:
            raise NotImplementedError
        print("cs cnt", pos_cnt, "ics cnt", neg_cnt)
        print("cs thr", triplets_sorted[-pos_cnt]["author_sim"])
        # print("ics thr", triplets_sorted_max[neg_cnt]["author_sim_max"])

        cur_ics_triplets_all = []
        ics_keys = set()
        for item in cur_ics_triplets + ics_triplets_others:
            aid = item["aid1"]
            pid = item["pid"]
            cur_key = aid + "~~~" + pid
            if cur_key not in ics_keys:
                cur_ics_triplets_all.append(item)
                ics_keys.add(cur_key)
        print("from", len(cur_ics_triplets), "to", len(cur_ics_triplets_all))
        cur_ics_triplets = cur_ics_triplets_all

        # print(cur_ics_triplets)
        # print(pid_err_set)
        ics_hit_err = {x["pid"] for x in cur_ics_triplets if x["pid"] in pid_err_set}
        ics_hit_eval = {x["pid"] for x in cur_ics_triplets if x["aid1"] in aids_eval}

        cs_hit_true = {x["pid"] for x in cur_cs_triplets if x["pid"] in pid_true_set}
        cs_hit_eval = {x["pid"] for x in cur_cs_triplets if x["aid1"] in aids_eval}
        err_rate = len(ics_hit_err)/len(ics_hit_eval)
        true_rate = len(cs_hit_true)/len(cs_hit_eval)
        print("********** true rate in cs", true_rate, len(cs_hit_true))
        print("********** err rate in ics", err_rate, len(ics_hit_err))

        utils.dump_json(cur_cs_triplets, file_dir, "cs_triplets_via_author_sim_{}.json".format(cur_suffix))
        utils.dump_json(cur_ics_triplets, file_dir, "ics_triplets_via_author_sim_{}.json".format(cur_suffix))


def gen_cs_and_ics_percentage_filter_uncommon():
    pos_percents = [0.1, 0.15, 0.2, 0.25, 0.3]
    neg_percents = [0.01, 0.03, 0.05, 0.07, 0.09]

    # pos_percents = [0.35, 0.4, 0.45, 0.5, 0.55]
    # neg_percents = [0.11, 0.13, 0.15, 0.17, 0.19]

    # triplets = utils.load_json(settings.OUT_DATASET_DIR, "triplets_train_author_sim_paper.json")
    if settings.data_source == "kddcup":
        triplets = utils.load_json(settings.OUT_DATASET_DIR, "triplets_train_author_sim_paper_more.json")
        name_aid_to_pids = utils.load_json(settings.DATASET_DIR, "train_author_pub_index_profile.json")
        paper_dict = utils.load_json(settings.DATASET_DIR, "paper_dict_used_mag.json")
    elif settings.data_source == "aminer":
        triplets = utils.load_json(settings.OUT_DATASET_DIR, "triplets_train_author_sim_paper.json")
        name_aid_to_pids = {}
        paper_dict = {}
    else:
        raise NotImplementedError

    aid_to_vid_cnt = dd(dict)
    aid_to_vid_filter = dd(dict)

    for name in tqdm(name_aid_to_pids):
        cur_name_dict = name_aid_to_pids[name]
        for aid in cur_name_dict:
            cur_pids = cur_name_dict[aid]
            for pid in cur_pids:
                pid_raw = pid.split("-")[0]
                cur_paper = paper_dict.get(pid_raw, {})
                venue_id = cur_paper.get("venue_id")
                if venue_id is not None and isinstance(venue_id, int):
                    if venue_id in aid_to_vid_cnt.get(aid, {}):
                        aid_to_vid_cnt[aid][venue_id] += 1
                    else:
                        aid_to_vid_cnt[aid][venue_id] = 1

    for aid in tqdm(aid_to_vid_cnt):
        cur_venue_dict = aid_to_vid_cnt[aid]
        for vid in cur_venue_dict:
            if cur_venue_dict[vid] > 1:
                aid_to_vid_filter[aid][vid] = cur_venue_dict[vid]

    n_triplets_all = len(triplets)
    triplets_sorted = sorted(triplets, key=lambda x: x["author_sim"])
    triplets_sorted_max = sorted(triplets, key=lambda x: x["author_sim_max"])

    valid_triplets = utils.load_json(settings.DATASET_DIR, "eval_na_checking_triplets_relabel1_valid.json")
    # test_triplets = utils.load_json(settings.DATASET_DIR, "eval_na_checking_triplets_relabel1_test.json")
    aids_eval = {x["aid1"] for x in valid_triplets}
    pid_err_set = {x["pid"] for x in valid_triplets if x["label"] == 0}
    pid_true_set = {x["pid"] for x in valid_triplets if x["label"] == 1}
    err_rate_in_eval = len(pid_err_set)/(len(pid_err_set) + len(pid_true_set))
    print("err_rate_in_eval", err_rate_in_eval)

    file_dir = join(settings.DATASET_DIR, "cs_ics")
    os.makedirs(file_dir, exist_ok=True)

    for i in range(len(pos_percents)):
        pos_cnt = int(n_triplets_all * pos_percents[i])
        neg_cnt = int(n_triplets_all * neg_percents[i])
        cur_cs_triplets = triplets_sorted[-pos_cnt:]
        if settings.data_source == "aminer":
            cur_ics_triplets = triplets_sorted_max[:neg_cnt]
            print("ics thr", triplets_sorted_max[neg_cnt]["author_sim_max"])
            cur_suffix = "pctpospminvf" + str(pos_percents[i])[0] + str(pos_percents[i])[2:] + "negpmax" + str(neg_percents[i])[0] + str(neg_percents[i])[2:]
        elif settings.data_source == "kddcup":
            cur_ics_triplets = triplets_sorted[:neg_cnt]
            print("ics thr", triplets_sorted[neg_cnt]["author_sim"])
            cur_suffix = "pctpospminvf" + str(pos_percents[i])[0] + str(pos_percents[i])[2:] + "negpmin" + str(neg_percents[i])[0] + str(neg_percents[i])[2:]
        else:
            raise NotImplementedError

        print("cs cnt", pos_cnt, "ics cnt", neg_cnt)
        print("cs thr", triplets_sorted[-pos_cnt]["author_sim"])
        # print("ics thr", triplets_sorted_max[neg_cnt]["author_sim_max"])

        cur_cs_triplets_filter = []
        for item in tqdm(cur_cs_triplets):
            pid = item["pid"]
            aid1 = item["aid1"]
            pid_raw = pid.split("-")[0]
            cur_paper = paper_dict.get(pid_raw, {})
            venue_id = cur_paper.get("venue_id")
            if venue_id and venue_id in aid_to_vid_filter.get(aid1, {}):
                cur_cs_triplets_filter.append(item)

        print("cs triplets filtered by common venues", len(cur_cs_triplets_filter))
        cur_cs_triplets = cur_cs_triplets_filter
        # print(cur_ics_triplets)
        # print(pid_err_set)
        ics_hit_err = {x["pid"] for x in cur_ics_triplets if x["pid"] in pid_err_set}
        ics_hit_eval = {x["pid"] for x in cur_ics_triplets if x["aid1"] in aids_eval}

        cs_hit_true = {x["pid"] for x in cur_cs_triplets if x["pid"] in pid_true_set}
        cs_hit_eval = {x["pid"] for x in cur_cs_triplets if x["aid1"] in aids_eval}
        err_rate = len(ics_hit_err)/len(ics_hit_eval)
        true_rate = len(cs_hit_true)/len(cs_hit_eval)
        print("********** true rate in cs", true_rate, len(cs_hit_true))
        print("********** err rate in ics", err_rate, len(ics_hit_err))

        utils.dump_json(cur_cs_triplets, file_dir, "cs_triplets_via_author_sim_{}.json".format(cur_suffix))
        utils.dump_json(cur_ics_triplets, file_dir, "ics_triplets_via_author_sim_{}.json".format(cur_suffix))


def divide_cs_and_ics_percentage():
    pos_percents = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    # triplets = utils.load_json(settings.OUT_DATASET_DIR, "triplets_train_author_sim_paper.json")
    triplets = utils.load_json(settings.OUT_DATASET_DIR, "triplets_train_author_sim_paper_more.json")
    n_triplets_all = len(triplets)
    triplets_sorted = sorted(triplets, key=lambda x: x["author_sim"])

    valid_triplets = utils.load_json(settings.DATASET_DIR, "eval_na_checking_triplets_relabel1_moreics_valid.json")
    # valid_triplets = utils.load_json(settings.DATASET_DIR, "eval_na_checking_triplets_relabel1_valid.json")
    # valid_triplets = utils.load_json(settings.DATASET_DIR, "eval_na_checking_triplets_relabel1_test.json")
    valid_pid_set = {x["pid"] for x in valid_triplets}
    pid_err_set = {x["pid"] for x in valid_triplets if x["label"] == 0}
    err_rate_in_eval = len(pid_err_set)/len(valid_triplets)
    print("err_rate_in_eval", err_rate_in_eval)
    file_dir = join(settings.DATASET_DIR, "cs_ics")

    for i in range(len(pos_percents)):
        pos_cnt = int(n_triplets_all * pos_percents[i])
        cur_cs_triplets = triplets_sorted[-pos_cnt:]
        cur_ics_triplets = triplets_sorted[: -pos_cnt]
        print("cs cnt", pos_cnt, "ics cnt", len(cur_ics_triplets))
        print("cs thr", triplets_sorted[-pos_cnt]["author_sim"])

        cs_hit_err = {x["pid"] for x in cur_cs_triplets if x["pid"] in pid_err_set}
        ics_hit_err = {x["pid"] for x in cur_ics_triplets if x["pid"] in pid_err_set}
        err_rate_cs = len(cs_hit_err)/len([x for x in cur_cs_triplets if x["pid"] in valid_pid_set])
        err_rate = len(ics_hit_err)/len([x for x in cur_ics_triplets if x["pid"] in valid_pid_set])
        print("********** err rate in cs", err_rate_cs, len(cs_hit_err))
        print("********** err rate in ics", err_rate, len(ics_hit_err))

        cur_suffix = "pctdiv" + str(pos_percents[i])[0] + str(pos_percents[i])[2:]
        utils.dump_json(cur_cs_triplets, file_dir, "cs_triplets_via_author_sim_{}.json".format(cur_suffix))
        utils.dump_json(cur_ics_triplets, file_dir, "ics_triplets_via_author_sim_{}.json".format(cur_suffix))


def divide_cs_and_ics_percentage_remove_ics_false():
    pos_percents = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    # triplets = utils.load_json(settings.OUT_DATASET_DIR, "triplets_train_author_sim_paper.json")
    triplets = utils.load_json(settings.OUT_DATASET_DIR, "triplets_train_author_sim_paper_more.json")
    n_triplets_all = len(triplets)
    triplets_sorted = sorted(triplets, key=lambda x: x["author_sim"])

    valid_triplets = utils.load_json(settings.DATASET_DIR, "eval_na_checking_triplets_relabel1_moreics_valid.json")
    # valid_triplets = utils.load_json(settings.DATASET_DIR, "eval_na_checking_triplets_relabel1_valid.json")
    # valid_triplets = utils.load_json(settings.DATASET_DIR, "eval_na_checking_triplets_relabel1_test.json")
    valid_pid_set = {x["pid"] for x in valid_triplets}
    pid_err_set = {x["pid"] for x in valid_triplets if x["label"] == 0}
    err_rate_in_eval = len(pid_err_set)/len(valid_triplets)
    print("err_rate_in_eval", err_rate_in_eval)
    file_dir = join(settings.DATASET_DIR, "cs_ics")

    for i in range(len(pos_percents)):
        pos_cnt = int(n_triplets_all * pos_percents[i])
        cur_cs_triplets = triplets_sorted[-pos_cnt:]
        cur_ics_triplets = triplets_sorted[: -pos_cnt]
        cur_ics_triplets = [x for x in cur_ics_triplets if x["label"] == 1]
        print("cs cnt", pos_cnt, "ics cnt", len(cur_ics_triplets))
        print("cs thr", triplets_sorted[-pos_cnt]["author_sim"])

        cs_hit_err = {x["pid"] for x in cur_cs_triplets if x["pid"] in pid_err_set}
        ics_hit_err = {x["pid"] for x in cur_ics_triplets if x["pid"] in pid_err_set}
        err_rate_cs = len(cs_hit_err)/len([x for x in cur_cs_triplets if x["pid"] in valid_pid_set])
        err_rate = len(ics_hit_err)/len([x for x in cur_ics_triplets if x["pid"] in valid_pid_set])
        print("********** err rate in cs", err_rate_cs, len(cs_hit_err))
        print("********** err rate in ics", err_rate, len(ics_hit_err))

        cur_suffix = "pctdiv" + str(pos_percents[i])[0] + str(pos_percents[i])[2:]
        utils.dump_json(cur_cs_triplets, file_dir, "cs_triplets_via_author_sim_{}.json".format(cur_suffix))
        utils.dump_json(cur_ics_triplets, file_dir, "ics_triplets_via_author_sim_{}.json".format(cur_suffix))


def gen_extra_triplets_via_author_sim_kddcup():
    file_dir = settings.DATASET_DIR
    aminer_name_aid_to_pids = utils.load_json(file_dir, "aminer_name_aid_to_pids.json")
    mag_name_aid_to_pids = utils.load_json(file_dir, "train_author_pub_index_profile.json")
    aminer_person_to_coauthors = utils.load_json(file_dir, "aminer_person_coauthors.json")
    mag_person_to_coauthors = utils.load_json(file_dir, "mag_person_coauthors.json")

    paper_dict = utils.load_json(file_dir, "paper_dict_used_mag.json")  # str: dict

    true_data = utils.load_json(file_dir, "true_data_map_filter.json")
    aids_set = set()
    for aid in true_data:
        aids_set.add(aid)

    aminer_name_pid_to_aid = dd(dict)
    for name in aminer_name_aid_to_pids:
        cur_name_dict = aminer_name_aid_to_pids[name]
        for aid in cur_name_dict:
            cur_pids = cur_name_dict[aid]
            for pid in cur_pids:
                aminer_name_pid_to_aid[name][pid] = aid

    valid_triplets = utils.load_json(settings.DATASET_DIR, "eval_na_checking_triplets_relabel1_moreics_valid.json")
    test_triplets = utils.load_json(settings.DATASET_DIR, "eval_na_checking_triplets_relabel1_moreics_test.json")
    eval_triplets = valid_triplets + test_triplets
    true_keys = set()
    false_keys = set()
    for item in eval_triplets:
        pid = item["pid"]
        aid1 = item["aid1"]
        cur_key = aid1 + "~~~" + str(pid)
        if item["label"] == 1:
            true_keys.add(cur_key)
        elif item["label"] == 0:
            false_keys.add(cur_key)

    pos_triplets = []
    neg_triplets = []
    n_paper_neg = 0
    n_venue_neg = 0
    n_coauthor_neg = 0
    n_map_by_others = 0

    pubs_overlap_thr = 0.2
    venues_overlap_thr = 0.4
    coauthors_overlap_thr = 0.3
    coauthors_overlap_thr_pos = 0.9
    pubs_overlap_thr_pos = 0.98
    venues_overlap_thr_pos = 0.95

    person_kddcup_id_to_aminer = utils.load_json(file_dir, "person_map_kddcup_to_aminer_oag.json")

    for i, name in enumerate(mag_name_aid_to_pids):
        logger.info("name %d: %s", i, name)
        cur_name_dict = mag_name_aid_to_pids[name]
        for aid in cur_name_dict:
            if aid not in aids_set:
                continue
            cur_pids = cur_name_dict[aid]
            cur_pids_int = [int(x.split("-")[0]) for x in cur_pids]
            vids_1 = [paper_dict[str(x)].get("venue_id") for x in cur_pids_int if
                      str(x) in paper_dict and "venue_id" in paper_dict[str(x)]]
            n_pubs_1 = len(cur_pids)

            coauthors_1 = mag_person_to_coauthors.get(aid, [])

            for pid in cur_pids:
                pid_int = int(pid.split("-")[0])
                if pid_int not in aminer_name_pid_to_aid.get(name, {}):
                    continue
                aid_map = aminer_name_pid_to_aid[name][pid_int]
                pids_aminer = aminer_name_aid_to_pids[name][aid_map]
                n_pubs_2 = len(pids_aminer)
                if n_pubs_1 < 5 or n_pubs_2 < 5:
                    continue

                vids_2 = [paper_dict[str(x)].get("venue_id") for x in pids_aminer if
                          str(x) in paper_dict and "venue_id" in paper_dict[str(x)]]

                # v_sim = utils.top_venue_sim(vids_1, vids_2)

                common_pids = set(cur_pids_int) & set(pids_aminer)
                n_common_pubs = len(common_pids)
                pubs_overlap_a = n_common_pubs / len(cur_pids_int)
                pubs_overlap_m = n_common_pubs / len(pids_aminer)

                coauthors_2 = aminer_person_to_coauthors.get(aid_map, [])
                coauthor_sim, _, _ = utils.top_coauthor_sim(coauthors_1, coauthors_2)

                cur_key = aid + "~~~" + str(pid)
                if cur_key in true_keys:
                    cur_truth = 1
                elif cur_key in false_keys:
                    cur_truth = 0
                else:
                    cur_truth = -1

                if aid in person_kddcup_id_to_aminer and person_kddcup_id_to_aminer[aid] == aid_map:
                # if True:
                    if coauthor_sim < coauthors_overlap_thr_pos:
                        continue
                    pos_triplets.append(
                        {"aid1": aid, "aid2": aid_map, "pid": pid, "name": name, "author_sim": coauthor_sim, "label": cur_truth})
                    # continue

                if False:
                    pass
                # elif pubs_overlap_a < pubs_overlap_thr and pubs_overlap_m < pubs_overlap_thr:
                #     neg_triplets.append({"aid1": aid, "aid2": aid_map, "pid": pid, "name": name,
                #                          "author_sim": min(pubs_overlap_a, pubs_overlap_m)})
                #     n_paper_neg += 1
                #     continue
                # elif v_sim < venues_overlap_thr:
                #     neg_triplets.append({"aid1": aid, "aid2": aid_map, "pid": pid, "name": name,
                #                          "author_sim": v_sim})
                #     n_venue_neg += 1
                #     continue
                elif coauthor_sim < coauthors_overlap_thr:
                    neg_triplets.append({"aid1": aid, "aid2": aid_map, "pid": pid, "name": name,
                                         "author_sim": coauthor_sim, "label": cur_truth})
                    n_coauthor_neg += 1
                    continue
                elif aid in person_kddcup_id_to_aminer and person_kddcup_id_to_aminer[aid] != aid_map:
                    if coauthor_sim >= 0.7:
                        continue
                    neg_triplets.append({"aid1": aid, "aid2": aid_map, "pid": pid, "name": name,
                                         "author_sim": coauthor_sim, "label": cur_truth})
                    n_map_by_others += 1

                # if False:
                #     pass
                # elif pubs_overlap_a >= pubs_overlap_thr_pos and pubs_overlap_m >= pubs_overlap_thr_pos:
                #     pos_triplets.append({"aid1": aid, "aid2": aid_map, "pid": pid, "name": name,
                #                          "author_sim": min(pubs_overlap_a, pubs_overlap_m)})
                # elif v_sim >= venues_overlap_thr_pos:
                #     pos_triplets.append({"aid1": aid, "aid2": aid_map, "pid": pid, "name": name,
                #                          "author_sim": v_sim})

    suffix = None

    logger.info("n_paper_neg %d, n_venue_neg %d, n_coauthors_neg %d, n_map_by_others %d",
                n_paper_neg, n_venue_neg, n_coauthor_neg, n_map_by_others)
    logger.info("number of postive triplets: %d", len(pos_triplets))
    utils.dump_json(pos_triplets, file_dir, "cs_triplets_via_author_sim_{}.json".format(suffix))
    logger.info("number of negative triplets: %d", len(neg_triplets))
    utils.dump_json(neg_triplets, file_dir, "ics_triplets_via_author_sim_{}.json".format(suffix))

    check_triplets_quality_kddcup(suffix)


def check_triplets_quality_kddcup(suffix):
    file_dir = settings.DATASET_DIR
    # valid_triplets = utils.load_json(file_dir, "eval_na_checking_pairs_valid_idx_with_clean_authors.json")
    # valid_triplets = utils.load_json(file_dir, "eval_na_checking_pairs_valid_idx_clean.json")
    valid_triplets = utils.load_json(file_dir, "eval_na_checking_triplets_relabel1_valid.json")
    # valid_triplets = utils.load_json(file_dir, "eval_na_checking_triplets_relabel1_test.json")
    # valid_triplets = utils.load_json(file_dir, "eval_na_checking_pairs_test_idx_clean.json")
    triplets_eval = valid_triplets

    pos_triplets = utils.load_json(file_dir, "cs_triplets_via_author_sim_{}.json".format(suffix))
    neg_triplets = utils.load_json(file_dir, "ics_triplets_via_author_sim_{}.json".format(suffix))
    pos_keys = set()
    neg_keys = set()

    for item in pos_triplets:
        aid1 = item["aid1"]
        pid = item["pid"]
        cur_key = aid1 + "~~~" + str(pid)
        pos_keys.add(cur_key)

    for item in neg_triplets:
        aid1 = item["aid1"]
        pid = item["pid"]
        cur_key = aid1 + "~~~" + str(pid)
        neg_keys.add(cur_key)

    y_pred = []
    y_true = []
    n_pred_0_true_1 = 0
    n_pred_1_true_0 = 0
    tt = 0
    ff = 0
    pred_true_cnt = 0
    real_true_cnt = 0
    pos_label_cnt = 0

    for item in triplets_eval:
        aid1 = item["aid1"]
        pid = item["pid"]
        cur_key = aid1 + "~~~" + str(pid)
        cur_label = item["label"]
        if cur_label == 1:
            pos_label_cnt += 1
        if cur_key in pos_keys:
            pred_true_cnt += 1
            y_pred.append(0)
            y_true.append(1 - cur_label)
            if cur_label == 0:
                n_pred_1_true_0 += 1
                # print("neg", item["name"], pub_aid, aid1, item["aid2"])
            else:
                tt += 1
                real_true_cnt += 1
        elif cur_key in neg_keys:
            y_pred.append(1)
            y_true.append(1 - cur_label)
            if cur_label == 1:
                n_pred_0_true_1 += 1
                # print("err", aid1, item["aid2"], pub_aid)
            else:
                ff += 1
                # print("find neg!", item["name"], pub_aid, aid1, item["aid2"])
        else:
            y_pred.append(0.5)
            y_true.append(cur_label)

    print(len(y_true), n_pred_0_true_1, n_pred_1_true_0, tt, ff)
    # prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    print("correct probs. real_true_cnt/pred_true_cnt", real_true_cnt/pred_true_cnt)
    print("positive label ratio", pos_label_cnt/len(triplets_eval))
    # prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    auc = roc_auc_score(y_true, y_pred)
    maps = average_precision_score(y_true, y_pred)
    # print(prec, rec, f1)
    print("prec", ff/(ff+n_pred_0_true_1))
    print("auc", auc, "maps", maps)


def gen_cs_neg_training_pairs_input_mat(suffix="pctpos03neg009"):
    pos_pairs = utils.load_json(settings.DATASET_DIR, "cs_triplets_via_author_sim_{}.json".format(suffix))

    neg_pairs = utils.load_json(settings.DATASET_DIR, "negative_paper_author_pairs_conna_mid.json")
    neg_key_to_pair = {x["pid"]: x for x in neg_pairs}
    neg_pairs_sample = []

    for pair in tqdm(pos_pairs):
        pid = pair["pid"]
        cur_pair = neg_key_to_pair[pid]
        cur_pair["aid1"] = cur_pair["aid"]
        neg_pairs_sample.append(cur_pair)

    pair_to_mat_in = utils.joblib_load_obj(settings.OUT_DATASET_DIR, "pa_pair_to_mat_mid_in.pkl")
    labels = [1] * len(pos_pairs) + [0] * len(neg_pairs_sample)

    attr_sim_mats = []
    empty_cnt = 0
    for pair in tqdm(pos_pairs + neg_pairs_sample):
        pid = pair["pid"]
        aid = pair["aid1"]
        cur_key_in = "{}~~~{}".format(aid, pid)
        if len(pair_to_mat_in[cur_key_in]) == 0:
            empty_cnt += 1
        attr_sim_mats.append(pair_to_mat_in[cur_key_in])
    print("empty cnt", empty_cnt)

    assert len(labels) == len(attr_sim_mats)

    utils.joblib_dump_obj(attr_sim_mats, settings.OUT_DATASET_DIR, "paper_author_matching_pairs_input_mat_mid_{}.pkl".format(suffix))
    utils.joblib_dump_obj(labels, settings.OUT_DATASET_DIR, "pa_labels_mid_{}.pkl".format(suffix))


def gen_cs_neg_training_pairs_portion():
    if settings.data_source == "kddcup":
        triplets = utils.load_json(settings.OUT_DATASET_DIR, "triplets_train_author_sim_paper_more.json")
        neg_pairs = utils.load_json(settings.DATASET_DIR, "negative_paper_author_pairs_conna_clean_1.json")
    elif settings.data_source == "aminer":
        triplets = utils.load_json(settings.OUT_DATASET_DIR, "triplets_train_author_sim_paper.json")
        neg_pairs = utils.load_json(settings.DATASET_DIR, "negative_paper_author_pairs_conna_mid.json")
    else:
        raise NotImplementedError
    n_triplets_all = len(triplets)
    pair_to_mat_in = utils.joblib_load_obj(settings.OUT_DATASET_DIR, "pa_pair_to_mat_mid_in.pkl")

    neg_key_to_pair = {x["pid"]: x for x in neg_pairs}

    pos_percents = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]
    triplets_sorted = sorted(triplets, key=lambda x: x["author_sim"])

    for i in range(len(pos_percents)):
        pos_cnt = int(n_triplets_all * pos_percents[i])
        cur_cs_triplets = triplets_sorted[-pos_cnt:]

        neg_pairs_sample = []

        for pair in tqdm(cur_cs_triplets):
            pid = pair["pid"]
            if pid not in neg_key_to_pair:
                continue
            cur_pair = neg_key_to_pair[pid]
            cur_pair["aid1"] = cur_pair["aid"]
            neg_pairs_sample.append(cur_pair)

        # for cur_pair in neg_pairs:
        #     cur_pair["aid1"] = cur_pair["aid"]
        #     neg_pairs_sample.append(cur_pair)

        print(len(cur_cs_triplets), len(neg_pairs_sample))

        labels = [1] * len(cur_cs_triplets) + [0] * len(neg_pairs_sample)

        attr_sim_mats = []
        empty_cnt = 0
        for pair in tqdm(cur_cs_triplets + neg_pairs_sample):
            pid = pair["pid"]
            aid = pair["aid1"]
            cur_key_in = "{}~~~{}".format(aid, pid)
            if len(pair_to_mat_in[cur_key_in]) == 0:
                empty_cnt += 1
            attr_sim_mats.append(pair_to_mat_in[cur_key_in])
        print("empty cnt", empty_cnt)

        # cur_suffix = "pctcstop" + str(pos_percents[i])[0] + str(pos_percents[i])[2:]
        cur_suffix = "pctcsnegtop" + str(pos_percents[i])[0] + str(pos_percents[i])[2:]

        assert len(labels) == len(attr_sim_mats)

        utils.joblib_dump_obj(attr_sim_mats, settings.OUT_DATASET_DIR,
                              "paper_author_matching_pairs_input_mat_mid_{}.pkl".format(cur_suffix))
        utils.joblib_dump_obj(labels, settings.OUT_DATASET_DIR, "pa_labels_mid_{}.pkl".format(cur_suffix))


def gen_training_paper_author_triplets():
    # check_dir = "/home/zfj/research-data/na-checking/kddcup/"
    # true_data = utils.load_json(check_dir, "true_data_map_filter.json")
    # aids_set = set()
    # for aid in true_data:
    #     aids_set.add(aid)

    valid_pairs = utils.load_json(settings.DATASET_DIR, "eval_na_checking_triplets_relabel1_moreics_valid.json")
    test_pairs = utils.load_json(settings.DATASET_DIR, "eval_na_checking_triplets_relabel1_moreics_test.json")
    aids_set = set()
    for item in valid_pairs + test_pairs:
        aids_set.add(item["aid1"])

    file_dir = settings.DATASET_DIR
    # name_aid_to_pids = utils.load_json(file_dir, "train_author_pub_index_profile.json")
    name_aid_to_pids = utils.load_json(file_dir, "train_author_pub_index_profile_enrich1.json")
    # name_aid_to_pids_out = utils.load_json(file_dir, "aminer_name_aid_to_pids.json")
    name_aid_to_pids_out = utils.load_json(file_dir, "aminer_name_aid_to_pids_with_idx_enrich1.json")
    paper_dict = utils.load_json(file_dir, "conna_pub_dict.json")

    name_pid_to_aid_out = dd(dict)
    for name in name_aid_to_pids_out:
        cur_name_dict = name_aid_to_pids_out[name]
        for aid in cur_name_dict:
            cur_pids = cur_name_dict[aid]
            for pid in cur_pids:
                name_pid_to_aid_out[name][int(pid.split("-")[0])] = aid

    triplets = []
    pos_pairs = []
    neg_pairs = []
    for i, name in enumerate(name_aid_to_pids):
        logger.info("name %d %s", i, name)
        cur_name_dict = name_aid_to_pids[name]
        cur_name_dict_filter = {}
        for aid in cur_name_dict:
            if len(cur_name_dict[aid]) >= 5:
                cur_name_dict_filter[aid] = cur_name_dict[aid]
        all_aids = list(cur_name_dict_filter.keys())
        # if len(all_aids) < 2:
        #     continue
        for j, aid in enumerate(cur_name_dict_filter):
            cur_pids = cur_name_dict_filter[aid]
            if aid.split("-")[0] not in aids_set:
                continue
            other_aids = list(set(all_aids) - {aid})
            for pid in cur_pids:
                pid_raw = pid.split("-")[0]

                if int(pid_raw) not in name_pid_to_aid_out.get(name, {}):
                    continue
                aid_map = name_pid_to_aid_out[name][int(pid_raw)]
                pids_out = name_aid_to_pids_out[name][aid_map]
                # if len(pids_out) < 5:  #TODO
                #     continue

                cur_authors = paper_dict.get(pid_raw, {}).get("authors", [])
                if len(cur_authors) > 300:
                    continue
                if len(cur_authors) > 30:
                    print("papers with many authors", pid_raw, len(cur_authors))
                if len(other_aids) > 0:
                    neg_aid = np.random.choice(other_aids)
                else:
                    neg_aid = np.random.choice(list(aids_set - {aid}))
                triplets.append({"pid": pid, "pos_aid": aid, "neg_aid": neg_aid, "name": name})
                pos_pairs.append({"pid": pid, "aid": aid, "name": name})
                neg_pairs.append({"pid": pid, "aid": neg_aid, "name": name})

    out_dir = settings.DATASET_DIR
    logger.info("number of triplets %d", len(triplets))
    utils.dump_json(triplets, out_dir, "paper_author_matching_triplets_more.json")
    utils.dump_json(pos_pairs, out_dir, "positive_paper_author_pairs_conna_more.json")
    utils.dump_json(neg_pairs, out_dir, "negative_paper_author_pairs_conna_more.json")


def tune_ics_thr_only(cs_thr=0.3):
    if settings.data_source == "kddcup":
        triplets = utils.load_json(settings.OUT_DATASET_DIR, "triplets_train_author_sim_paper_more.json")
    elif settings.data_source == "aminer":
        triplets = utils.load_json(settings.OUT_DATASET_DIR, "triplets_train_author_sim_paper.json")
    else:
        raise NotImplementedError
    n_triplets_all = len(triplets)

    valid_triplets = utils.load_json(settings.DATASET_DIR, "eval_na_checking_triplets_relabel1_valid.json")
    # valid_triplets = utils.load_json(settings.DATASET_DIR, "eval_na_checking_triplets_relabel1_test.json")
    pid_err_set = {x["pid"] for x in valid_triplets if x["label"] == 0}

    triplets_sorted = sorted(triplets, key=lambda x: x["author_sim"])
    triplets_sorted_max = sorted(triplets, key=lambda x: x["author_sim_max"])
    pos_cnt = int(n_triplets_all * cs_thr)
    cur_cs_triplets = triplets_sorted[-pos_cnt:]

    neg_percents = [0.01, 0.03, 0.05, 0.07, 0.09, 0.13, 0.17]
    file_dir = join(settings.DATASET_DIR, "cs_ics")
    os.makedirs(file_dir, exist_ok=True)

    for i in range(len(neg_percents)):
        cur_ics_cnt = int(n_triplets_all * neg_percents[i])
        # cur_ics_triplets = triplets_sorted[: cur_ics_cnt]
        cur_ics_triplets = triplets_sorted_max[: cur_ics_cnt]

        ics_hit_err = {x["pid"] for x in cur_ics_triplets if x["pid"] in pid_err_set}
        err_rate = len(ics_hit_err)/len(cur_ics_triplets)
        print("********** err rate", err_rate, len(ics_hit_err))

        cur_suffix = "pctcspmin" + str(cs_thr)[0] + str(cs_thr)[2:] + "icspmax" + str(neg_percents[i])[0] + str(neg_percents[i])[2:]
        # cur_suffix = "pctcspmin" + str(cs_thr)[0] + str(cs_thr)[2:] + "icspmin" + str(neg_percents[i])[0] + str(neg_percents[i])[2:]

        utils.dump_json(cur_cs_triplets, file_dir, "cs_triplets_via_author_sim_{}.json".format(cur_suffix))
        utils.dump_json(cur_ics_triplets, file_dir, "ics_triplets_via_author_sim_{}.json".format(cur_suffix))


def test_ics_intuition_gen_triplets(cur_suffix="pctpos015neg003"):
    pos_pairs = utils.load_json(join(settings.DATASET_DIR, "cs_ics"), "cs_triplets_via_author_sim_{}.json".format(cur_suffix))
    neg_pairs = utils.load_json(join(settings.DATASET_DIR, "cs_ics"), "ics_triplets_via_author_sim_{}.json".format(cur_suffix))
    print("processing before", len(pos_pairs), len(neg_pairs))

    valid_triplets = utils.load_json(settings.DATASET_DIR, "eval_na_checking_triplets_relabel1_valid.json")
    test_triplets = utils.load_json(settings.DATASET_DIR, "eval_na_checking_triplets_relabel1_test.json")

    name_aid_to_pids_in = utils.load_json(settings.DATASET_DIR, "train_author_pub_index_profile.json")
    name_aid_to_pids_out = utils.load_json(settings.DATASET_DIR, "aminer_name_aid_to_pids.json")

    pids_key_neg = set()

    add_cnt = 0
    remove_cnt = 0
    for item in tqdm(valid_triplets + test_triplets):
        label = item["label"]
        name = item["name"]
        aid1 = item["aid1"]
        aid2 = item["aid2"]

        if label == 1:
            continue
        add_cnt += 1
        pids_key_neg.add(item["pid"])
        cur_pids_in = name_aid_to_pids_in[name][aid1]
        cur_pids_out = name_aid_to_pids_out[name][aid2]
        p_sim_min, p_sim_max = utils.paper_overlap_ratio(cur_pids_in, cur_pids_out)
        item["author_sim"] = p_sim_min
        neg_pairs.append(item)

    pos_pairs_new = []
    for item in pos_pairs:
        if item["pid"] not in pids_key_neg:
            pos_pairs_new.append(item)
        else:
            remove_cnt += 1

    print("add cnt", add_cnt, "remove cnt", remove_cnt)
    print("processing after", len(pos_pairs_new), len(neg_pairs))
    cur_suffix_new = cur_suffix + "realneg"
    utils.dump_json(pos_pairs_new, join(settings.DATASET_DIR, "cs_ics"), "cs_triplets_via_author_sim_{}.json".format(cur_suffix_new))
    utils.dump_json(neg_pairs, join(settings.DATASET_DIR, "cs_ics"), "ics_triplets_via_author_sim_{}.json".format(cur_suffix_new))


def try_gen_perfect_cs_and_ics():
    valid_triplets = utils.load_json(settings.DATASET_DIR, "eval_na_checking_triplets_relabel1_valid.json")
    test_triplets = utils.load_json(settings.DATASET_DIR, "eval_na_checking_triplets_relabel1_test.json")

    name_aid_to_pids_in = utils.load_json(settings.DATASET_DIR, "train_author_pub_index_profile.json")
    name_aid_to_pids_out = utils.load_json(settings.DATASET_DIR, "aminer_name_aid_to_pids.json")

    pos_pairs = []
    neg_pairs = []

    for item in tqdm(valid_triplets + test_triplets):
        label = item["label"]
        name = item["name"]
        aid1 = item["aid1"]
        aid2 = item["aid2"]

        cur_pids_in = name_aid_to_pids_in[name][aid1]
        cur_pids_out = name_aid_to_pids_out[name][aid2]
        p_sim_min, p_sim_max = utils.paper_overlap_ratio(cur_pids_in, cur_pids_out)
        item["author_sim"] = p_sim_min

        if label == 1:
            pos_pairs.append(item)
        else:
            neg_pairs.append(item)

    print("processing after", len(pos_pairs), len(neg_pairs))
    cur_suffix_new = "perfect"
    utils.dump_json(pos_pairs, join(settings.DATASET_DIR, "cs_ics"), "cs_triplets_via_author_sim_{}.json".format(cur_suffix_new))
    utils.dump_json(neg_pairs, join(settings.DATASET_DIR, "cs_ics"), "ics_triplets_via_author_sim_{}.json".format(cur_suffix_new))


def gen_eval_triplets_relabel_to_label(role="valid"):
    eval_pairs = utils.load_json(settings.DATASET_DIR, "eval_na_checking_triplets_relabel1_{}.json".format(role))
    file_dir = settings.DATASET_DIR

    paper_dict = utils.load_json(file_dir, "paper_dict_used_mag.json")  # str: dict
    an = addressNormalization()

    mag_aid_to_aff = utils.load_json(file_dir, "mag_person_aff.json")
    aminer_aid_to_aff = utils.load_json(file_dir, "aminer_person_aff.json")

    pairs_with_aff = []
    for item in tqdm(eval_pairs):
        pid = item["pid"]
        pid_raw, a_idx = pid.split("-")
        a_idx = int(a_idx)
        d_new = dd(dict)
        d_new["paper"] = {"pid": item["pid"], "name": item["name"]}
        d_new["mag"] = {"aid1": item["aid1"], "aff": mag_aid_to_aff.get(str(item["aid1"]), "")}
        d_new["mag"]["url"] = "http://166.111.5.162:20412/mag_author/" + str(item["aid1"])
        d_new["aminer"] = {"aid2": item["aid2"], "aff": aminer_aid_to_aff.get(item["aid2"], "")}
        d_new["aminer"]["url"] = "https://www.aminer.cn/profile/" + item["aid2"]
        d_new["labels"] = {"original": item["label"], "zfj": ""}
        if pid_raw in paper_dict:
            cur_paper_dict = paper_dict[pid_raw]
            cur_title = cur_paper_dict.get("title", "")
            d_new["paper"]["title"] = cur_title
            cur_authors = cur_paper_dict.get("authors", [])
            for a_i, a in enumerate(cur_authors):
                if a_i == a_idx:
                    cur_aff = a.get("OriginalAffiliation", "")
                    if cur_aff is not None and len(cur_aff) > 0:
                        d_new["paper"]["aff"] = cur_aff.strip()
                    else:
                        d_new["paper"]["aff"] = ""
        else:
            d_new["paper"]["aff"] = ""

        if len(d_new["paper"]["aff"]) > 0:
            _, aff_main = an.find_inst(d_new["paper"]["aff"].upper().replace(".", ""))
            d_new["paper"]["aff_main"] = aff_main.lower()
        else:
            d_new["paper"]["aff_main"] = ""

        pairs_with_aff.append(d_new)

    utils.dump_json(pairs_with_aff, settings.DATASET_DIR, "eval_triplets_to_label_{}.json".format(role))


def gen_aff_sim_ics_triplets(thr=0.05):
    valid_pairs = utils.load_json(settings.DATASET_DIR, "eval_triplets_to_label_valid.json")
    test_pairs = utils.load_json(settings.DATASET_DIR, "eval_triplets_to_label_test.json")

    pid_err_set = {x["paper"]["pid"] for x in valid_pairs + test_pairs if x["labels"]["original"] == 0}

    an = addressNormalization()

    ics_triplets_aff = []
    for pair in tqdm(valid_pairs + test_pairs):
        paper_aff = pair.get("paper", {}).get("aff_main")
        person_out_aff = pair.get("aminer", {}).get("aff")
        if paper_aff is None or person_out_aff is None:
            continue
        _, aff_main = an.find_inst(person_out_aff.upper().replace(".", ""))
        if len(paper_aff) < 7 or len(person_out_aff) < 7:
            continue
        person_out_aff = aff_main.lower()
        cur_aff_sim = utils.aff_sim_ngrams(paper_aff, person_out_aff)
        if cur_aff_sim < thr:
            pair_new = {}
            pair_new["author_sim"] = cur_aff_sim
            pair_new["pid"] = pair["paper"]["pid"]
            pair_new["aid1"] = pair["mag"]["aid1"]
            pair_new["aid2"] = pair["aminer"]["aid2"]
            pair_new["name"] = pair["paper"]["name"]
            # pair.pop("label")
            ics_triplets_aff.append(pair_new)

    ics_hit_err = {x["pid"] for x in ics_triplets_aff if x["pid"] in pid_err_set}
    err_rate = len(ics_hit_err) / len(ics_triplets_aff)
    print("********** err rate", err_rate, len(ics_hit_err))

    utils.dump_json(ics_triplets_aff, join(settings.DATASET_DIR, "cs_ics"), "ics_triplets_via_author_sim_aff{}.json".format(thr))


def analyze_cs_ics_false_partition(suffix="pctpospmin015negpmin003"):
    pos_pairs = utils.load_json(join(settings.DATASET_DIR, "cs_ics"), "cs_triplets_via_author_sim_{}.json".format(suffix))
    neg_pairs = utils.load_json(join(settings.DATASET_DIR, "cs_ics"), "ics_triplets_via_author_sim_{}.json".format(suffix))
    valid_triplets = utils.load_json(settings.DATASET_DIR, "eval_na_checking_triplets_relabel1_valid.json")
    aids_eval = {x["aid1"] for x in valid_triplets}
    pid_err_set = {x["pid"] for x in valid_triplets if x["label"] == 0}
    pid_true_set = {x["pid"] for x in valid_triplets if x["label"] == 1}

    for pair in tqdm(pos_pairs):
        pid = pair["pid"]
        if pid in pid_err_set:
            print("-----------------------------------")
            print(pair)


def gen_eval_triplets_one_to_many(role="valid"):
    eval_triplets = utils.load_json(settings.DATASET_DIR, "eval_na_checking_triplets_relabel1_{}.json".format(role))
    aid_in_to_out = dd(set)
    aid_out_to_in = dd(set)
    for item in tqdm(eval_triplets):
        aid1 = item["aid1"]
        aid2 = item["aid2"]
        aid_in_to_out[aid1].add(aid2)
        aid_out_to_in[aid2].add(aid1)

    eval_triplets_filter = []
    persons_filter = set()
    for item in tqdm(eval_triplets):
        aid1 = item["aid1"]
        aid2 = item["aid2"]
        person_map_cnt = aid_in_to_out[aid1]
        if len(person_map_cnt) > 1 or len(aid_out_to_in[aid2]) > 1:
            eval_triplets_filter.append(item)
            persons_filter.add(aid1)

    print(len(eval_triplets), len(aid_in_to_out), len(eval_triplets_filter), len(persons_filter))
    utils.dump_json(eval_triplets_filter, settings.DATASET_DIR, "eval_na_checking_triplets_relabel1_one2many_{}.json".format(role))


def cal_train_eval_person_overlap():
    name_aid_to_pids = utils.load_json(settings.DATASET_DIR, "train_author_pub_index_profile.json")
    name_aid_to_pids_out = utils.load_json(settings.DATASET_DIR, "aminer_name_aid_to_pids_with_idx.json")
    labeled_data = utils.load_json(settings.DATASET_DIR, "true_data_map_filter.json")
    paper_dict = utils.load_json(settings.DATASET_DIR, 'conna_pub_dict.json')

    aid_to_outlier = dd(set)
    for aid in tqdm(labeled_data):
        cur_pids_err = labeled_data[aid]["outliers"]
        aid_to_outlier[aid] = {str(x) for x in cur_pids_err}

    name_pid_to_aid_out = dd(dict)
    for name in name_aid_to_pids_out:
        cur_name_dict = name_aid_to_pids_out[name]
        for aid in cur_name_dict:
            cur_pids = cur_name_dict[aid]
            for pid in cur_pids:
                name_pid_to_aid_out[name][pid] = aid

    n_persons = 0
    n_persons_onetomany = 0
    n_pubs = 0
    aid_to_pid_err_to_map = dd(list)
    n_pubs_to_label = 0
    for name in tqdm(name_aid_to_pids):
        cur_name_dict = name_aid_to_pids[name]
        for aid in cur_name_dict:
            cur_pids = cur_name_dict[aid]
            cur_pid_to_pid_with_idx = {x.split("-")[0]: x for x in cur_pids}
            pids_err_map = aid_to_outlier.get(aid, set())

            cur_map_aids_out = set()
            pids_has_map = []
            for pid in cur_pids:
                aid_map = name_pid_to_aid_out.get(name, {}).get(pid)
                if aid_map:
                    cur_map_aids_out.add(aid_map)
                    pids_has_map.append(pid)

            cur_pids_raw = {x.split("-")[0] for x in pids_has_map}
            common_pids = cur_pids_raw & pids_err_map
            if len(common_pids) > 0:
                n_persons += 1
            if len(cur_map_aids_out) > 1:
                n_persons_onetomany += 1
                n_pubs += len(pids_has_map)
                if len(common_pids) == 0:
                    print("pids_err_map", pids_err_map)
                    print("cur_pids_raw", cur_pids_raw)
                    if len(pids_err_map) > 0 and len(cur_pids_raw) > len(pids_err_map) + 1:
                        for pid in pids_err_map:
                            if pid not in paper_dict:
                                continue
                            aid_to_pid_err_to_map[aid].append(
                                {"pid": pid, "pidx": cur_pid_to_pid_with_idx.get(pid, ""),
                                 "title": paper_dict[pid].get("title"), "name": name, "aid2": "",
                                 "candidates": list(cur_map_aids_out)
                                 })
                            n_pubs_to_label += 1

    print("n_persons", n_persons)
    print("n_persons_onetomany", n_persons_onetomany)
    print("n_pubs", n_pubs)
    print("n_pubs_to_label", n_pubs_to_label)
    print("n_persons to label", len(aid_to_pid_err_to_map))

    utils.dump_json(aid_to_pid_err_to_map, settings.DATASET_DIR, "aid_to_pid_err_to_map_label.json")


def get_paper_id_to_aminer_person_chunk(i_chunk=0, bs=5):
    # aid_to_pid_err_to_map = utils.load_json(settings.DATASET_DIR, "aid_to_pid_err_to_map_label_1.json")
    # aid_to_pid_err_to_map = utils.load_json(settings.DATASET_DIR, "aid_to_pid_err_to_map_label_auto6.json")
    aid_to_pid_err_to_map = utils.load_json(settings.DATASET_DIR, "aid_to_pid_err_nonhit_1to1_to_label_1.json")
    client = MongoDBClientKexie()
    aminer_pub_col = client.aminer_pub_col

    add_cnt1 = 0
    add_cnt2 = 0
    aids = sorted(list(aid_to_pid_err_to_map.keys()))
    aids_chunk = aids[bs*i_chunk: bs*(i_chunk+1)]
    cur_dict = dd(list)
    for aid in tqdm(aids_chunk):
        cur_papers = aid_to_pid_err_to_map[aid]
        for item in cur_papers:
            if item["aid2"] and item["pidx"]:
                continue
            title = item["title"]
            pid_aminer = get_paper_id_from_title(title)
            if pid_aminer is None:
                continue
            cur_paper = aminer_pub_col.find_one({"_id": ObjectId(pid_aminer)})
            if cur_paper is None:
                continue
            logger.info("pid found %s, add cnt %d, %d", pid_aminer, add_cnt1, add_cnt2)
            cur_name = item["name"]
            cur_authors = cur_paper.get("authors", [])
            if item["pidx"]:
                a_idx = int(item["pidx"].split("-")[-1])
                author_map = cur_authors[a_idx]
                if "_id" in author_map:
                    item["aid2"] = str(author_map["_id"])
                    add_cnt2 += 1
                print(cur_name, item["pidx"], item["aid2"], author_map)
            else:
                cur_authors_new = []
                for ai, a in enumerate(cur_authors):
                    d = {}
                    d["name"] = utils.clean_name(a.get("name", ""))
                    if "_id" in a:
                        d["_id"] = str(a["_id"])
                    d["a_idx"] = ai
                    d["name_sim"] = fuzz.ratio(d["name"], cur_name)
                    cur_authors_new.append(d)
                cur_authors_sorted = sorted(cur_authors_new, key=lambda x: x["name_sim"], reverse=True)
                print("sorted", cur_authors_sorted)
                if len(cur_authors_sorted) > 0 and cur_authors_sorted[0]["name_sim"] >= 60:
                    item["pidx"] = item["pid"] + "-" + str(cur_authors_sorted[0]["a_idx"])
                    add_cnt1 += 1
                    if "_id" in cur_authors_sorted[0]:
                        item["aid2"] = str(cur_authors_sorted[0]["_id"])
                        add_cnt2 += 1
                print(cur_name, item["pidx"], item["aid2"], cur_authors_sorted[0])
            cur_dict[aid].append(item)
            time.sleep(10)
            # if add_cnt2 >= 10:
            #     break

    out_dir = join(settings.DATASET_DIR, "outlier-label-chunk")
    os.makedirs(out_dir, exist_ok=True)
    # utils.dump_json(aid_to_pid_err_to_map, settings.DATASET_DIR, "aid_to_pid_err_to_map_label_auto7.json")
    utils.dump_json(cur_dict, out_dir, "aid_to_pid_err_nonhit_1to1_to_label_auto{}.json".format(i_chunk))


def get_paper_id_to_aminer_person_all(bs=5):
    aid_to_pid_err_to_map = utils.load_json(settings.DATASET_DIR, "aid_to_pid_err_nonhit_1to1_to_label_1.json")
    aids = sorted(list(aid_to_pid_err_to_map.keys()))
    n_chunks = int(np.ceil(len(aids)/bs))
    logger.info("n_chunks %d", n_chunks)
    for i in range(29):
        get_paper_id_to_aminer_person_chunk(i, bs)


def get_pid_err_non_hit_to_label():
    name_aid_to_pids = utils.load_json(settings.DATASET_DIR, "train_author_pub_index_profile.json")
    name_aid_to_pids_out = utils.load_json(settings.DATASET_DIR, "aminer_name_aid_to_pids_with_idx.json")
    labeled_data = utils.load_json(settings.DATASET_DIR, "true_data_map_filter.json")
    paper_dict = utils.load_json(settings.DATASET_DIR, 'conna_pub_dict.json')

    aid_to_outlier = dd(set)
    for aid in tqdm(labeled_data):
        cur_pids_err = labeled_data[aid]["outliers"]
        aid_to_outlier[aid] = {str(x) for x in cur_pids_err}

    name_pid_to_aid_out = dd(dict)
    for name in name_aid_to_pids_out:
        cur_name_dict = name_aid_to_pids_out[name]
        for aid in cur_name_dict:
            cur_pids = cur_name_dict[aid]
            for pid in cur_pids:
                name_pid_to_aid_out[name][pid] = aid

    n_persons_add = 0
    n_pubs_to_label = 0
    aid_to_pid_err_to_map = dd(list)
    for name in tqdm(name_aid_to_pids):
        cur_name_dict = name_aid_to_pids[name]
        for aid in cur_name_dict:
            cur_pids = cur_name_dict[aid]
            cur_pid_to_pid_with_idx = {x.split("-")[0]: x for x in cur_pids}
            pids_err_map = aid_to_outlier.get(aid, set())

            cur_map_aids_out = set()
            pids_has_map = []
            for pid in cur_pids:
                aid_map = name_pid_to_aid_out.get(name, {}).get(pid)
                if aid_map:
                    cur_map_aids_out.add(aid_map)
                    pids_has_map.append(pid)

            cur_pids_raw = {x.split("-")[0] for x in pids_has_map}
            common_pids = cur_pids_raw & pids_err_map

            if len(common_pids) == 0 and len(cur_map_aids_out) == 1:
                if len(cur_pids_raw) < len(pids_err_map) + 1:
                    continue
                if len(pids_err_map) < 3:
                    continue
                n_persons_add += 1
                for pid in pids_err_map:
                    if pid not in paper_dict:
                        continue
                    aid_to_pid_err_to_map[aid].append(
                        {"pid": pid, "pidx": cur_pid_to_pid_with_idx.get(pid, ""),
                         "title": paper_dict[pid].get("title"), "name": name, "aid2": "",
                         "candidates": list(cur_map_aids_out)
                         })
                    n_pubs_to_label += 1

    print("n_pubs_to_label", n_pubs_to_label)
    print("n_persons to label", len(aid_to_pid_err_to_map))

    utils.dump_json(aid_to_pid_err_to_map, settings.DATASET_DIR, "aid_to_pid_err_nonhit_1to1_to_label.json")


def gen_kddcup_more_ics_eval_data():
    in_dir = join(settings.DATASET_DIR, "outlier-label-chunk")
    files = []
    for f in os.listdir(in_dir):
        if f.startswith("aid_to_pid_err_nonhit_1to1_to_label_auto"):
            files.append(f)

    d = {}
    n_pubs = 0
    for f in tqdm(files):
        cur_dict = utils.load_json(in_dir, f)
        for aid in cur_dict:
            cur_pids_err = cur_dict[aid]
            cur_aid_out_to_cnt = dd(int)
            aid_out_main = cur_pids_err[0]["candidates"][0]
            for item in cur_pids_err:
                if item["aid2"]:
                    cur_aid_out_to_cnt[item["aid2"]] += 1
            cur_pids_err_filter = []
            for item in cur_pids_err:
                if item["aid2"] and cur_aid_out_to_cnt[item["aid2"]] >= 3 and item["aid2"] != aid_out_main:
                    cur_pids_err_filter.append(item)
            if len(cur_pids_err_filter) >= 3:
                d[aid] = cur_pids_err_filter
                n_pubs += len(cur_pids_err_filter)

    print("n_persons", len(d))
    print("n_pids_err", n_pubs)

    utils.dump_json(d, settings.DATASET_DIR, "aid_to_pid_err_merge_one_to_many_filter.json")


def gen_aid_to_pid_err_one_to_many_all():
    d = utils.load_json(settings.DATASET_DIR, "aid_to_pid_err_merge_one_to_many_filter.json")
    labeled_data = utils.load_json(settings.DATASET_DIR, "true_data_map_filter.json")
    name_aid_to_pids = utils.load_json(settings.DATASET_DIR, "train_author_pub_index_profile.json")
    name_aid_to_pids_out = utils.load_json(settings.DATASET_DIR, "aminer_name_aid_to_pids_with_idx.json")

    aid_to_outlier = dd(set)
    for aid in tqdm(labeled_data):
        cur_pids_err = labeled_data[aid]["outliers"]
        aid_to_outlier[aid] = {str(x) for x in cur_pids_err}

    name_pid_to_aid_out = dd(dict)
    for name in name_aid_to_pids_out:
        cur_name_dict = name_aid_to_pids_out[name]
        for aid in cur_name_dict:
            cur_pids = cur_name_dict[aid]
            for pid in cur_pids:
                name_pid_to_aid_out[name][str(pid)] = aid

    n_pubs_add = 0
    for name in tqdm(name_aid_to_pids):
        cur_name_dict = name_aid_to_pids[name]
        for aid in cur_name_dict:
            cur_pids = cur_name_dict[aid]
            cur_pid_to_pid_with_idx = {x.split("-")[0]: x for x in cur_pids}
            pids_err_map = aid_to_outlier.get(aid, set())

            cur_map_aids_out = set()
            pids_has_map = []
            for pid in cur_pids:
                aid_map = name_pid_to_aid_out.get(name, {}).get(pid)
                if aid_map:
                    cur_map_aids_out.add(aid_map)
                    pids_has_map.append(pid)
            # print("cur_map_aids_out", cur_map_aids_out)

            cur_pids_raw = {x.split("-")[0] for x in pids_has_map}
            common_pids = cur_pids_raw & pids_err_map

            if len(common_pids) > 0 and len(cur_map_aids_out) > 1:
                d[aid] = []
                for pid in pids_err_map:
                    if pid not in cur_pid_to_pid_with_idx:
                        continue
                    pidx = cur_pid_to_pid_with_idx[pid]
                    if pidx not in name_pid_to_aid_out.get(name, {}):
                        continue
                    aid2 = name_pid_to_aid_out[name][pidx]
                    cur_item = {"pid": pid, "pidx": pidx, "name": name, "aid2": aid2}
                    d[aid].append(cur_item)
                    n_pubs_add += 1

    print("n_persons", len(d))
    print("n_pids_err", n_pubs_add)

    utils.dump_json(d, settings.DATASET_DIR, "aid_to_pid_err_merge_one_to_many_filter_all.json")


def enrich_eval_triplets_kddcup():
    valid_pairs = utils.load_json(settings.DATASET_DIR, "eval_na_checking_triplets_relabel1_valid.json")
    test_pairs = utils.load_json(settings.DATASET_DIR, "eval_na_checking_triplets_relabel1_test.json")
    name_aid_to_pids = utils.load_json(settings.DATASET_DIR, "train_author_pub_index_profile.json")
    name_aid_to_pids_out = utils.load_json(settings.DATASET_DIR, "aminer_name_aid_to_pids_with_idx.json")
    labeled_data = utils.load_json(settings.DATASET_DIR, "true_data_map_filter.json")

    d = utils.load_json(settings.DATASET_DIR, "aid_to_pid_err_merge_one_to_many_filter_all.json")
    eval_keys = set()
    valid_pairs_new = []
    test_pairs_new = []
    aids_used = set()

    pid_to_aid_out_2 = {}
    for aid in tqdm(d):
        for item in d[aid]:
            pid_to_aid_out_2[item["pidx"]] = item["aid2"]

    name_pid_to_aid_out = dd(dict)
    for name in name_aid_to_pids_out:
        cur_name_dict = name_aid_to_pids_out[name]
        for aid in cur_name_dict:
            cur_pids = cur_name_dict[aid]
            for pid in cur_pids:
                name_pid_to_aid_out[name][str(pid)] = aid

    for item in tqdm(valid_pairs):
        aid1 = item["aid1"]
        pid = item["pid"]
        cur_key = aid1 + "###" + pid
        if cur_key not in eval_keys:
            valid_pairs_new.append(item)
            eval_keys.add(cur_key)
            aids_used.add(aid1)

    for item in tqdm(test_pairs):
        aid1 = item["aid1"]
        pid = item["pid"]
        cur_key = aid1 + "###" + pid
        if cur_key not in eval_keys:
            test_pairs_new.append(item)
            eval_keys.add(cur_key)
            aids_used.add(aid1)
    print("aids used", len(aids_used))
    print("eval keys before", len(eval_keys))

    aid_to_pids_true = dd(set)
    aid_to_pids_false = dd(set)
    n_hit_used = 0
    for name in tqdm(name_aid_to_pids):
        cur_name_dict = name_aid_to_pids[name]
        for aid in cur_name_dict:
            if aid not in d:
                continue
            if aid in aids_used:
                n_hit_used += 1
                continue
            cur_pids_idx = set(cur_name_dict[aid])
            cur_pids_true = set(str(x) for x in labeled_data[aid]["normals"])
            cur_pids_false = set(str(x) for x in labeled_data[aid]["outliers"])
            for pid in cur_pids_idx:
                # if pid not in name_pid_to_aid_out.get(name, {}):
                #     continue
                pid_raw = pid.split("-")[0]
                if pid_raw in cur_pids_true:
                    aid_to_pids_true[aid].add(pid)
                elif pid_raw in cur_pids_false:
                    aid_to_pids_false[aid].add(pid)

            for item in d[aid]:
                pid = item["pidx"]
                aid_to_pids_false[aid].add(pid)
            print("cur person", aid, len(aid_to_pids_true[aid]), len(aid_to_pids_false[aid]))

    print("n_hit used", n_hit_used)
    aids_sorted = sorted(aid_to_pids_true.keys(), key=lambda x: len(aid_to_pids_true[x]) + len(aid_to_pids_false.get(x, set())), reverse=True)
    aids_valid_add = set()
    aids_test_add = set()
    print("aids add", len(aids_sorted))

    for i, aid in enumerate(aids_sorted):
        if i % 2 == 0:
            aids_valid_add.add(aid)
        else:
            aids_test_add.add(aid)

    for name in tqdm(name_aid_to_pids):
        cur_name_dict = name_aid_to_pids[name]
        for aid in cur_name_dict:
            if aid in aids_valid_add:
                cur_list = valid_pairs_new
            elif aid in aids_test_add:
                cur_list = test_pairs_new
            else:
                continue
            cur_pids_idx = aid_to_pids_true.get(aid) | aid_to_pids_false.get(aid)
            for pid in cur_pids_idx:
                if pid in name_pid_to_aid_out.get(name, {}):
                    aid_map = name_pid_to_aid_out[name][pid]
                elif pid in pid_to_aid_out_2:
                    aid_map = pid_to_aid_out_2[pid]
                else:
                    continue
                if pid in aid_to_pids_true.get(aid):
                    label = 1
                elif pid in aid_to_pids_false.get(aid):
                    label = 0
                else:
                    continue
                cur_item = {"pid": pid, "name": name, "aid1": aid, "aid2": aid_map, "label": label}
                cur_key = aid + "###" + pid
                if cur_key not in eval_keys:
                    cur_list.append(cur_item)
                    eval_keys.add(cur_key)

    print("valid", len(valid_pairs_new), sum(1-x["label"] for x in valid_pairs_new))
    print("test", len(test_pairs_new), sum(1-x["label"] for x in test_pairs_new))
    print("eval keys after", len(eval_keys))

    utils.dump_json(valid_pairs_new, settings.DATASET_DIR, "eval_na_checking_triplets_relabel1_moreics_valid.json")
    utils.dump_json(test_pairs_new, settings.DATASET_DIR, "eval_na_checking_triplets_relabel1_moreics_test.json")


def enrich_name_aid_to_pids_kddcup():
    name_aid_to_pids = utils.load_json(settings.DATASET_DIR, "train_author_pub_index_profile.json")
    name_aid_to_pids_out = utils.load_json(settings.DATASET_DIR, "aminer_name_aid_to_pids_with_idx.json")

    valid_pairs = utils.load_json(settings.DATASET_DIR, "eval_na_checking_triplets_relabel1_moreics_valid.json")
    test_pairs = utils.load_json(settings.DATASET_DIR, "eval_na_checking_triplets_relabel1_moreics_test.json")

    n_hit_in = 0
    n_hit_out = 0
    for item in tqdm(valid_pairs + test_pairs):
        pid = item["pid"]
        aid1 = item["aid1"]
        aid2 = item["aid2"]
        name = item["name"]
        cur_pids_in = set(name_aid_to_pids.get(name, {}).get(aid1, []))
        if pid not in cur_pids_in:
            name_aid_to_pids[name][aid1].append(pid)
            n_hit_in += 1
        cur_pids_out = set(name_aid_to_pids_out.get(name, {}).get(aid2, []))
        if pid not in cur_pids_out:
            if aid2 in name_aid_to_pids_out.get(name, {}):
                name_aid_to_pids_out[name][aid2].append(pid)
            else:
                name_aid_to_pids_out[name][aid2] = [pid]
            n_hit_out += 1
    print("n_hit_in", n_hit_in, "n_hit_out", n_hit_out)

    name_aid_to_pids_new = dd(dict)
    name_aid_to_pids_out_new = dd(dict)
    for name in tqdm(name_aid_to_pids):
        cur_name_dict = name_aid_to_pids[name]
        for aid in cur_name_dict:
            name_aid_to_pids_new[name][aid] = list(set(name_aid_to_pids[name][aid]))

    for name in tqdm(name_aid_to_pids_out):
        cur_name_dict = name_aid_to_pids_out[name]
        for aid in cur_name_dict:
            name_aid_to_pids_out_new[name][aid] = list(set(name_aid_to_pids_out[name][aid]))

    utils.dump_json(name_aid_to_pids_new, settings.DATASET_DIR, "train_author_pub_index_profile_enrich1.json")
    utils.dump_json(name_aid_to_pids_out_new, settings.DATASET_DIR, "aminer_name_aid_to_pids_with_idx_enrich1.json")


def gen_training_pairs_input_mat_kddcup():
    file_dir = settings.DATASET_DIR
    print("file dir", file_dir)
    name_aid_to_pids = utils.load_json(file_dir, "train_author_pub_index_profile.json")

    pos_pairs = utils.load_json(file_dir, "positive_paper_author_pairs_conna_clean_1.json")
    neg_pairs = utils.load_json(file_dir, "negative_paper_author_pairs_conna_clean_1.json")
    paper_dict = utils.load_json(file_dir, "paper_dict_used_mag.json")  # str: dict

    emb_dir = "/home/zfj/research-data/conna/kddcup/emb-228/"
    word_emb_mat = utils.load_data(emb_dir, "word_emb.array")
    # author_emb_mat = utils.load_data(emb_dir, "author_emb.array")
    pub_feature_dict = utils.load_data(emb_dir, "pub_feature.ids")

    pairs = pos_pairs + neg_pairs
    labels = [1] * len(pos_pairs) + [0] * len(neg_pairs)

    attr_sim_mats = []
    valid_cnt = 0

    for i, pair in enumerate(pairs):
        if i % 100 == 0:
            logger.info("pair %d, valid cnt %d", i, valid_cnt)
            # if i > 0 and len(attr_sim_mats[-1]) > 0:
            #     print(attr_sim_mats_dot[-1][0])

        pid = pair["pid"]
        aid = pair["aid"]
        name = pair["name"]
        cur_author_pubs = name_aid_to_pids[name][aid]
        cur_pids = [x for x in cur_author_pubs if x != pid]
        cur_paper_year = paper_dict.get(str(pid.split("-")[0]), {}).get("year", 2022)  # this line works str(pid)
        # print("cur_pids_before", cur_pids)
        if len(cur_pids) <= 10:
            pids_selected = cur_pids
        else:
            papers_attr = [(x, paper_dict[str(x.split("-")[0])]) for x in cur_pids if str(x.split("-")[0]) in paper_dict]
            papers_sorted = sorted(papers_attr, key=lambda x: abs(cur_paper_year - x[1].get("year", 2022)))
            pids_selected = [x[0] for x in papers_sorted][:10]

        cur_mats = []
        flag = False

        if pid not in pub_feature_dict:
            attr_sim_mats.append(cur_mats)
            continue

        author_id_list, author_idf_list, word_id_list, word_idf_list = pub_feature_dict[pid]

        p_embs = word_emb_mat[word_id_list[: settings.MAX_MAT_SIZE]]

        if len(p_embs) == 0:
            attr_sim_mats.append(cur_mats)
            continue

        for ap in pids_selected:
            if ap not in pub_feature_dict:
                # cur_mats.append(np.zeros(shape=(settings.MAX_MAT_SIZE, settings.MAX_MAT_SIZE), dtype=np.float16))
                continue
            ap_author_ids, _, ap_word_ids, _ = pub_feature_dict[ap]
            ap_embs = word_emb_mat[ap_word_ids[: settings.MAX_MAT_SIZE]]

            if len(ap_embs) == 0:
                # cur_mats.append(
                #     np.zeros(shape=(settings.MAX_MAT_SIZE, settings.MAX_MAT_SIZE), dtype=np.float16))
                pass
            else:
                cur_sim = cosine_similarity(p_embs, ap_embs)
                cur_mats.append(cur_sim)
                flag = True

        if flag:
            valid_cnt += 1
        else:
            print(pid, aid, name, pids_selected)

        attr_sim_mats.append(cur_mats)

        if i >= settings.TEST_SIZE - 1:
            break

    labels = labels[: settings.TEST_SIZE]
    print(len(attr_sim_mats), len(labels))

    print(len(attr_sim_mats), len(labels))
    assert len(attr_sim_mats) == len(labels)
    utils.joblib_dump_obj(attr_sim_mats, settings.OUT_DATASET_DIR, "paper_author_matching_pairs_input_mat_mid.pkl")
    utils.joblib_dump_obj(labels, settings.OUT_DATASET_DIR, "pa_labels_mid.pkl")


def gen_eval_pair_input_mat_kddcup(role="valid"):
    file_dir = settings.DATASET_DIR
    print("file dir", file_dir)
    name_aid_to_pids = utils.load_json(file_dir, "train_author_pub_index_profile.json")
    name_aid_to_pids_out = utils.load_json(file_dir, "aminer_name_aid_to_pids_with_idx.json")

    pairs = utils.load_json(file_dir, "eval_na_checking_triplets_relabel1_{}.json".format(role))

    paper_dict = utils.load_json(file_dir, "paper_dict_used_mag.json")  # str: dict

    emb_dir = "/home/zfj/research-data/conna/kddcup/emb-228/"
    word_emb_mat = utils.load_data(emb_dir, "word_emb.array")
    # author_emb_mat = utils.load_data(file_dir, "author_emb.array")
    pub_feature_dict = utils.load_data(emb_dir, "pub_feature.ids")

    attr_sim_mats = []
    sim_mats_2 = []
    valid_cnt = 0
    valid_cnt2 = 0

    for i, pair in enumerate(pairs):
        if i % 100 == 0:
            logger.info("pair %d, valid cnt %d", i, valid_cnt)
            if i > 0 and len(attr_sim_mats[-1]) > 0:
                # print(attr_sim_mats[-1][0])
                pass

        pid = pair["pid"]
        aid = pair["aid1"]
        aid2 = pair["aid2"]
        name = pair["name"]
        cur_author_pubs = name_aid_to_pids[name][aid]
        cur_pids = [x for x in cur_author_pubs if x != pid]
        cur_paper_year = paper_dict.get(str(pid.split("-")[0]), {}).get("year", 2022)  # this line works str(pid)
        # print("cur_pids_before", cur_pids)
        if len(cur_pids) <= 10:
            pids_selected = cur_pids
        else:
            papers_attr = [(x, paper_dict[str(x.split("-")[0])]) for x in cur_pids if str(x.split("-")[0]) in paper_dict]
            papers_sorted = sorted(papers_attr, key=lambda x: abs(cur_paper_year - x[1].get("year", 2022)))
            pids_selected = [x[0] for x in papers_sorted][:10]

        cur_mats = []
        flag = False
        flag2 = False

        if pid not in pub_feature_dict:
            attr_sim_mats.append(cur_mats)
            continue

        author_id_list, author_idf_list, word_id_list, word_idf_list = pub_feature_dict[pid]

        p_embs = word_emb_mat[word_id_list[: settings.MAX_MAT_SIZE]]

        if len(p_embs) == 0:
            attr_sim_mats.append(cur_mats)
            continue

        for ap in pids_selected:
            if ap not in pub_feature_dict:
                # cur_mats.append(np.zeros(shape=(settings.MAX_MAT_SIZE, settings.MAX_MAT_SIZE), dtype=np.float16))
                continue
            ap_author_ids, _, ap_word_ids, _ = pub_feature_dict[ap]
            ap_embs = word_emb_mat[ap_word_ids[: settings.MAX_MAT_SIZE]]

            if len(ap_embs) == 0:
                # cur_mats.append(
                #     np.zeros(shape=(settings.MAX_MAT_SIZE, settings.MAX_MAT_SIZE), dtype=np.float16))
                pass
            else:
                cur_sim = cosine_similarity(p_embs, ap_embs)
                cur_mats.append(cur_sim)
                flag = True

        if flag:
            valid_cnt += 1
        else:
            print(pid, aid, name, pids_selected)

        attr_sim_mats.append(cur_mats)

        # out
        cur_mats_out = []
        cur_author_pubs2 = name_aid_to_pids_out[name][str(aid2)]
        cur_pids2 = [x for x in cur_author_pubs2 if x != pid]
        if len(cur_pids2) <= 10:
            pids_selected2 = cur_pids2
        else:
            papers_attr2 = [(x, paper_dict[str(x.split("-")[0])]) for x in cur_pids2 if
                            str(x.split("-")[0]) in paper_dict]
            papers_sorted2 = sorted(papers_attr2, key=lambda x: abs(cur_paper_year - x[1].get("year", 2022)))
            pids_selected2 = [x[0] for x in papers_sorted2][:10]

        for ap in pids_selected2:
            if ap not in pub_feature_dict:
                continue
            ap_author_ids, _, ap_word_ids, _ = pub_feature_dict[ap]
            ap_embs = word_emb_mat[ap_word_ids[: settings.MAX_MAT_SIZE]]
            if len(ap_embs) > 0:
                cur_sim = cosine_similarity(p_embs, ap_embs)
                cur_mats_out.append(cur_sim)
                flag2 = True

        sim_mats_2.append(cur_mats_out)

        if flag2:
            valid_cnt2 += 1
        else:
            print(pid, aid2, name, pids_selected2)

        if i % 100 == 0:
            logger.info("pair %d, valid cnt1 %d, valid cnt2 %d", i, valid_cnt, valid_cnt2)

        if i >= settings.TEST_SIZE - 1:
            break

    assert len(attr_sim_mats) == len(sim_mats_2)

    print("number of mats", len(attr_sim_mats))
    utils.joblib_dump_obj(attr_sim_mats, settings.OUT_DATASET_DIR, "paper_author_matching_input_mat_eval_mid_{}.pkl".format(role))
    utils.joblib_dump_obj(sim_mats_2, settings.OUT_DATASET_DIR, "paper_author_matching_input_mat_eval_mid_out_{}.pkl".format(role))


def gen_cs_ics_input_mat_kddcup():
    file_dir = settings.DATASET_DIR
    print("file dir", file_dir)
    name_aid_to_pids_out = utils.load_json(file_dir, "aminer_name_aid_to_pids_with_idx.json")
    pos_pairs = utils.load_json(file_dir, "cs_triplets_via_author_sim_None.json")
    neg_pairs = utils.load_json(file_dir, "ics_triplets_via_author_sim_None.json")

    neg_pairs_copy = deepcopy(neg_pairs)
    dup_times = len(pos_pairs) // len(neg_pairs_copy)
    logger.info("dup times %d", dup_times)
    for _ in range(1, dup_times):
        neg_pairs += neg_pairs_copy

    logger.info("n_pos pairs %d, n_neg pairs %d", len(pos_pairs), len(neg_pairs))

    n_pos = min(len(pos_pairs), 10*len(neg_pairs))
    if len(pos_pairs) <= 10 * len(neg_pairs):
        pos_pairs_sample = pos_pairs
    else:
        pos_pairs_sample = np.random.choice(pos_pairs, n_pos, replace=False)
    pairs = list(pos_pairs_sample) + neg_pairs
    logger.info("number of pairs %d", len(pairs))
    labels = [1] * len(pos_pairs_sample) + [0] * len(neg_pairs)
    author_sim_scores = [x["author_sim"] for x in pos_pairs_sample] + [x["author_sim"] for x in neg_pairs]
    truths = [x["label"] for x in pos_pairs_sample] + [x["label"] for x in neg_pairs]

    name_aid_to_pids = utils.load_json(file_dir, "train_author_pub_index_profile.json")
    paper_dict = utils.load_json(file_dir, "paper_dict_used_mag.json")  # str: dict

    emb_dir = "/home/zfj/research-data/conna/kddcup/emb-228/"
    word_emb_mat = utils.load_data(emb_dir, "word_emb.array")
    pub_feature_dict = utils.load_data(emb_dir, "pub_feature.ids")

    mats_in = []
    mats_out = []
    valid_cnt1 = 0
    valid_cnt2 = 0

    for i, pair in enumerate(pairs):
        pid = pair["pid"]
        aid1 = pair["aid1"]
        aid2 = pair["aid2"]
        name = pair["name"]

        cur_author_pubs = name_aid_to_pids[name][aid1]
        cur_pids = [x for x in cur_author_pubs if x != pid]
        cur_paper_year = paper_dict.get(str(pid.split("-")[0]), {}).get("year", 2022)  # this line works str(pid)

        if len(cur_pids) <= 10:
            pids_selected = cur_pids
        else:
            papers_attr = [(x, paper_dict[str(x.split("-")[0])]) for x in cur_pids if str(x.split("-")[0]) in paper_dict]
            papers_sorted = sorted(papers_attr, key=lambda x: abs(cur_paper_year - x[1].get("year", 2022)))
            pids_selected = [x[0] for x in papers_sorted][:10]

        cur_mats_in = []
        cur_mats_out = []
        flag1 = False
        flag2 = False

        if pid not in pub_feature_dict:
            mats_in.append(cur_mats_in)
            mats_out.append(cur_mats_out)
            continue

        author_id_list, author_idf_list, word_id_list, word_idf_list = pub_feature_dict[pid]

        p_embs = word_emb_mat[word_id_list[: settings.MAX_MAT_SIZE]]

        for ap in pids_selected:
            if ap not in pub_feature_dict:
                # cur_mats.append(np.zeros(shape=(settings.MAX_MAT_SIZE, settings.MAX_MAT_SIZE), dtype=np.float16))
                continue
            ap_author_ids, _, ap_word_ids, _ = pub_feature_dict[ap]
            ap_embs = word_emb_mat[ap_word_ids[: settings.MAX_MAT_SIZE]]
            if len(ap_embs) > 0:
                cur_sim = cosine_similarity(p_embs, ap_embs)
                cur_mats_in.append(cur_sim)
                flag1 = True

        mats_in.append(cur_mats_in)
        if flag1:
            valid_cnt1 += 1
        else:
            print(pid, aid1, name, pids_selected)

        # out
        cur_author_pubs2 = name_aid_to_pids_out[name][str(aid2)]
        cur_pids2 = [x for x in cur_author_pubs2 if x != pid]
        if len(cur_pids2) <= 10:
            pids_selected2 = cur_pids2
        else:
            papers_attr2 = [(x, paper_dict[str(x.split("-")[0])]) for x in cur_pids2 if str(x.split("-")[0]) in paper_dict]
            papers_sorted2 = sorted(papers_attr2, key=lambda x: abs(cur_paper_year - x[1].get("year", 2022)))
            pids_selected2 = [x[0] for x in papers_sorted2][:10]

        for ap in pids_selected2:
            if ap not in pub_feature_dict:
                continue
            ap_author_ids, _, ap_word_ids, _ = pub_feature_dict[ap]
            ap_embs = word_emb_mat[ap_word_ids[: settings.MAX_MAT_SIZE]]
            if len(ap_embs) > 0:
                cur_sim = cosine_similarity(p_embs, ap_embs)
                cur_mats_out.append(cur_sim)
                flag2 = True

        mats_out.append(cur_mats_out)
        if flag2:
            valid_cnt2 += 1
        else:
            print(pid, aid2, name, pids_selected2)

        if i % 100 == 0:
            logger.info("pair %d, valid cnt1 %d, valid cnt2 %d", i, valid_cnt1, valid_cnt2)

        if i >= settings.TEST_SIZE - 1:
            break

    file_dir = settings.OUT_DATASET_DIR
    labels = labels[: settings.TEST_SIZE]
    assert len(mats_in) == len(mats_out) == len(labels)
    utils.joblib_dump_obj(mats_in, file_dir, "cs_and_ics_train_input_mat_in_mid_None.pkl")
    utils.joblib_dump_obj(mats_out, file_dir, "cs_and_ics_train_input_mat_out_mid_None.pkl")
    utils.joblib_dump_obj(labels, file_dir, "cs_and_ics_train_labels_mid_None.pkl")
    utils.joblib_dump_obj(author_sim_scores, file_dir, "cs_and_ics_train_author_sim_scores_mid_None.pkl")
    utils.joblib_dump_obj(truths, file_dir, "cs_and_ics_train_truths_mid_None.pkl")


def gen_fuzzy_neg_pairs_kddcup():
    file_dir = settings.DATASET_DIR
    neg_triplets = utils.load_json(file_dir, "cs_triplets_via_author_sim_None.json")
    pos_triplets = utils.load_json(file_dir, "ics_triplets_via_author_sim_None.json")

    pos_pairs = utils.load_json(file_dir, "positive_paper_author_pairs_conna_clean_1.json")
    neg_pairs = utils.load_json(file_dir, "negative_paper_author_pairs_conna_clean_1.json")

    used_keys = set()

    for item in pos_triplets:
        aid1 = item["aid1"]
        pid = item["pid"]
        cur_key = aid1 + "~~~" + str(pid)
        if cur_key not in used_keys:
            used_keys.add(cur_key)
            # pass

    for item in neg_triplets:
        aid1 = item["aid1"]
        pid = item["pid"]
        cur_key = aid1 + "~~~" + str(pid)
        if cur_key not in used_keys:
            used_keys.add(cur_key)

    pos_triplets = []
    neg_triplets = []

    name_pid_to_aid_in = dd(dict)

    mag_person_to_coauthors = utils.load_json(file_dir, "mag_person_coauthors.json")

    for pair in pos_pairs:
        aid = pair["aid"]
        pid = pair["pid"]
        name = pair["name"]
        cur_key = aid + "~~~" + str(pid)
        name_pid_to_aid_in[name][pid] = aid
        if cur_key not in used_keys:
            used_keys.add(cur_key)
            aid2 = aid
            pos_triplets.append({"aid1": aid, "aid2": aid2, "pid": pid, "name": name, "author_sim": 1})

    for pair in neg_pairs:
        aid = pair["aid"]
        pid = pair["pid"]
        name = pair["name"]
        cur_key = aid + "~~~" + str(pid)
        if cur_key not in used_keys:
            used_keys.add(cur_key)
            aid2 = name_pid_to_aid_in[name][pid]
            coauthors_1 = mag_person_to_coauthors.get(aid, [])
            coauthors_2 = mag_person_to_coauthors.get(aid2, [])
            coauthor_sim, _, _ = utils.top_coauthor_sim(coauthors_1, coauthors_2)
            neg_triplets.append({"aid1": aid, "aid2": aid2, "pid": pid, "name": name, "author_sim": coauthor_sim})

    # utils.dump_json(pos_triplets, file_dir, "positive_triplets_manual.json")
    # utils.dump_json(neg_triplets, file_dir, "negative_triplets_manual.json")
    utils.dump_json(pos_triplets, file_dir, "fuzzy_positive_remain_triplets.json")
    utils.dump_json(neg_triplets, file_dir, "negative_remain_triplets.json")


def gen_fuzzy_neg_input_mat_kddcup():
    file_dir = settings.DATASET_DIR
    print("file dir", file_dir)

    pos_pairs = utils.load_json(file_dir, "fuzzy_positive_remain_triplets.json")
    neg_pairs = utils.load_json(file_dir, "negative_remain_triplets.json")

    pairs = list(pos_pairs) + neg_pairs
    logger.info("number of pairs %d", len(pairs))
    labels = [1] * len(pos_pairs) + [0] * len(neg_pairs)
    # author_sim_scores = [x["author_sim"] for x in pos_pairs] + [x["author_sim"] for x in neg_pairs]

    name_aid_to_pids = utils.load_json(file_dir, "train_author_pub_index_profile.json")
    paper_dict = utils.load_json(file_dir, "paper_dict_used_mag.json")  # str: dict

    emb_dir = "/home/zfj/research-data/conna/kddcup/emb-228/"
    word_emb_mat = utils.load_data(emb_dir, "word_emb.array")
    pub_feature_dict = utils.load_data(emb_dir, "pub_feature.ids")

    mats_in = []
    mats_aux = []
    valid_cnt1 = 0
    valid_cnt2 = 0

    for i, pair in enumerate(pairs):
        pid = pair["pid"]
        aid1 = pair["aid1"]
        aid2 = pair["aid2"]
        name = pair["name"]

        cur_author_pubs = name_aid_to_pids[name][aid1]
        cur_pids = [x for x in cur_author_pubs if x != pid]
        cur_paper_year = paper_dict.get(str(pid.split("-")[0]), {}).get("year", 2022)  # this line works str(pid)

        if len(cur_pids) <= 10:
            pids_selected = cur_pids
        else:
            papers_attr = [(x, paper_dict[str(x.split("-")[0])]) for x in cur_pids if str(x.split("-")[0]) in paper_dict]
            papers_sorted = sorted(papers_attr, key=lambda x: abs(cur_paper_year - x[1].get("year", 2022)))
            pids_selected = [x[0] for x in papers_sorted][:10]

        cur_mats_in = []
        cur_mats_out = []
        flag1 = False
        flag2 = False

        if pid not in pub_feature_dict:
            mats_in.append(cur_mats_in)
            mats_aux.append(cur_mats_out)
            continue

        author_id_list, author_idf_list, word_id_list, word_idf_list = pub_feature_dict[pid]

        p_embs = word_emb_mat[word_id_list[: settings.MAX_MAT_SIZE]]

        for ap in pids_selected:
            if ap not in pub_feature_dict:
                # cur_mats.append(np.zeros(shape=(settings.MAX_MAT_SIZE, settings.MAX_MAT_SIZE), dtype=np.float16))
                continue
            ap_author_ids, _, ap_word_ids, _ = pub_feature_dict[ap]
            ap_embs = word_emb_mat[ap_word_ids[: settings.MAX_MAT_SIZE]]
            if len(ap_embs) > 0:
                cur_sim = cosine_similarity(p_embs, ap_embs)
                cur_mats_in.append(cur_sim)
                flag1 = True

        mats_in.append(cur_mats_in)
        if flag1:
            valid_cnt1 += 1
        else:
            print(pid, aid1, name, pids_selected)

        # out
        cur_author_pubs2 = name_aid_to_pids[name][str(aid2)]
        cur_pids2 = [x for x in cur_author_pubs2 if x != pid]
        if len(cur_pids2) <= 10:
            pids_selected2 = cur_pids2
        else:
            pids_selected2 = list(np.random.choice(cur_pids2, 10, replace=False))

        for ap in pids_selected2:
            if ap not in pub_feature_dict:
                continue
            ap_author_ids, _, ap_word_ids, _ = pub_feature_dict[ap]
            ap_embs = word_emb_mat[ap_word_ids[: settings.MAX_MAT_SIZE]]
            if len(ap_embs) > 0:
                cur_sim = cosine_similarity(p_embs, ap_embs)
                cur_mats_out.append(cur_sim)
                flag2 = True

        mats_aux.append(cur_mats_out)
        if flag2:
            valid_cnt2 += 1
        else:
            print(pid, aid2, name, pids_selected2)

        if i % 100 == 0:
            logger.info("pair %d, valid cnt1 %d, valid cnt2 %d", i, valid_cnt1, valid_cnt2)

        if i >= settings.TEST_SIZE - 1:
            break

    file_dir = settings.OUT_DATASET_DIR
    labels = labels[: settings.TEST_SIZE]
    assert len(mats_in) == len(mats_aux) == len(labels)
    utils.joblib_dump_obj(mats_in, file_dir, "pa_fuzzy_neg_input_mat_in_mid_None.pkl")
    # utils.joblib_dump_obj(mats_aux, file_dir, "paper_author_matching_matrices_aux_manual.pkl")
    utils.joblib_dump_obj(labels, file_dir, "fuzzy_neg_labels_mid_None.pkl")
    # utils.joblib_dump_obj(author_sim_scores, file_dir, "author_sim_scores_manual.pkl")


def gen_ics_triplets_for_eval_whoiswho(suffix="pctpospmin03negpmax012"):
    ics_triplets = utils.load_json(join(settings.DATASET_DIR, "cs_ics"), "ics_triplets_via_author_sim_{}.json".format(suffix))

    valid_triplets = utils.load_json(settings.DATASET_DIR, "eval_na_checking_triplets_relabel1_valid.json")
    test_triplets = utils.load_json(settings.DATASET_DIR, "eval_na_checking_triplets_relabel1_test.json")

    ics_keys = set()
    for item in ics_triplets:
        aid = item["aid1"]
        pid = item["pid"]
        cur_key = aid + "~~~" + pid
        ics_keys.add(cur_key)

    valid_triplets_new = []
    test_triplets_new = []

    for item in valid_triplets:
        aid = item["aid1"]
        pid = item["pid"]
        cur_key = aid + "~~~" + pid
        if cur_key in ics_keys:
            valid_triplets_new.append(item)

    for item in test_triplets:
        aid = item["aid1"]
        pid = item["pid"]
        cur_key = aid + "~~~" + pid
        if cur_key in ics_keys:
            test_triplets_new.append(item)

    print("num valid ics", len(valid_triplets_new))
    print("num test ics", len(test_triplets_new))

    utils.dump_json(valid_triplets_new, settings.DATASET_DIR, "ics_triplets_relabel1_valid_{}.json".format(suffix))
    utils.dump_json(test_triplets_new, settings.DATASET_DIR, "ics_triplets_relabel1_test_{}.json".format(suffix))


def gen_ics_input_mat_mid_whoiswho(role="valid", suffix="pctpospmin03negpmax012"):
    file_dir = settings.DATASET_DIR

    ics_triplets = utils.load_json(file_dir, "ics_triplets_relabel1_{}_{}.json".format(role, suffix))
    pair_to_mat_in = utils.joblib_load_obj(settings.OUT_DATASET_DIR, "pa_pair_to_mat_mid_in.pkl")
    pair_to_mat_out = utils.joblib_load_obj(settings.OUT_DATASET_DIR, "pa_pair_to_mat_mid_out.pkl")

    attr_sim_mats = []
    sim_mats_2 = []

    for pair in tqdm(ics_triplets):
        pid = pair["pid"]
        aid = pair["aid1"]
        aid2 = pair["aid2"]

        cur_key_in = "{}~~~{}".format(aid, pid)
        cur_key_out = "{}~~~{}".format(aid2, pid)

        attr_sim_mats.append(pair_to_mat_in[cur_key_in])
        sim_mats_2.append(pair_to_mat_out[cur_key_out])

    print("number of mats", len(attr_sim_mats))
    utils.joblib_dump_obj(attr_sim_mats, settings.OUT_DATASET_DIR, "ics_input_mat_mid_{}_{}.pkl".format(role, suffix))
    utils.joblib_dump_obj(sim_mats_2, settings.OUT_DATASET_DIR, "ics_input_mat_mid_out_{}_{}.pkl".format(role, suffix))


def try_gen_extra_ics_triplets_for_whoiswho(thr=0.1):
    oag_linking_dir = join(settings.DATA_DIR, "..", "oag-2-1")
    aperson_to_mperson = load_oag_linking_pairs(oag_linking_dir, "author_linking_pairs_2020.txt")
    triplets = utils.load_json(settings.OUT_DATASET_DIR, "triplets_train_author_sim_paper.json")

    triplets_sorted = sorted(triplets, key=lambda x: x["author_sim"])

    valid_triplets = utils.load_json(settings.DATASET_DIR, "eval_na_checking_triplets_relabel1_valid.json")
    aids_eval = {x["aid1"] for x in valid_triplets}
    pid_err_set = {x["pid"] for x in valid_triplets if x["label"] == 0}
    pid_true_set = {x["pid"] for x in valid_triplets if x["label"] == 1}
    err_rate_in_eval = len(pid_err_set)/(len(pid_err_set) + len(pid_true_set))
    print("err_rate_in_eval", err_rate_in_eval)

    ics_triplets_others = []
    # thr = 0.1
    for item in tqdm(triplets_sorted):
        aid1 = item["aid1"]
        aid2 = int(item["aid2"])
        if aid1 in aperson_to_mperson and int(aperson_to_mperson[aid1]) != aid2:
            if item["author_sim"] < thr:
                ics_triplets_others.append(item)

    ics_hit_err = {x["pid"] for x in ics_triplets_others if x["pid"] in pid_err_set}
    ics_hit_eval = {x["pid"] for x in ics_triplets_others if x["aid1"] in aids_eval}
    err_rate = len(ics_hit_err) / len(ics_hit_eval)

    print("********** err rate in ics", err_rate, len(ics_hit_err))

    file_dir = join(settings.DATASET_DIR, "cs_ics")
    os.makedirs(file_dir, exist_ok=True)
    print(len(ics_triplets_others))

    utils.dump_json(ics_triplets_others, file_dir, "ics_triplets_via_author_sim_extra_others_{}.json".format(thr))


def gen_extra_triplets_via_author_sim_whoiswho_new_4():
    file_dir = settings.DATASET_DIR
    aminer_name_aid_to_pids = utils.load_json(file_dir, "name_aid_to_pids_in_mid_filter.json")
    mag_name_aid_to_pids = utils.load_json(file_dir, "name_aid_to_pids_out_mid_filter.json")
    oag_linking_dir = join(settings.DATA_DIR, "..", "oag-2-1")
    aperson_to_mperson = load_oag_linking_pairs(oag_linking_dir, "author_linking_pairs_2020.txt")
    paper_dict = utils.load_json(file_dir, "paper_dict_used_mag_mid.json")  # str: dict
    aminer_person_to_coauthors = utils.load_json(file_dir, "person_coauthors_in_mid.json")
    mag_person_to_coauthors = utils.load_json(file_dir, "person_coauthors_out_mid.json")

    # name_to_aids = utils.load_json(file_dir, "aminer_name_to_aids.json")
    # aids_set = set()
    # for name in name_to_aids:
    #     cur_aids = set(name_to_aids[name])
    #     aids_set |= cur_aids

    valid_pairs = utils.load_json(settings.DATASET_DIR, "eval_na_checking_triplets_relabel1_valid.json")
    test_pairs = utils.load_json(settings.DATASET_DIR, "eval_na_checking_triplets_relabel1_test.json")
    aids_set = {x["aid1"] for x in valid_pairs + test_pairs}

    eval_triplets = valid_pairs + test_pairs
    true_keys = set()
    false_keys = set()
    for item in eval_triplets:
        pid = item["pid"]
        aid1 = item["aid1"]
        cur_key = aid1 + "~~~" + str(pid)
        if item["label"] == 1:
            true_keys.add(cur_key)
        elif item["label"] == 0:
            false_keys.add(cur_key)

    mag_name_pid_to_aid = dd(dict)
    for name in mag_name_aid_to_pids:
        cur_name_dict = mag_name_aid_to_pids[name]
        for aid in cur_name_dict:
            cur_pids = cur_name_dict[aid]
            for pid in cur_pids:
                mag_name_pid_to_aid[name][pid] = aid

    # pos_pairs = utils.load_json(file_dir, "positive_paper_author_pairs_conna_clean.json")
    pos_pairs = utils.load_json(file_dir, "positive_paper_author_pairs_conna_mid.json")
    aids_train = {item["aid"] for item in pos_pairs}

    pos_triplets = []
    neg_triplets = []
    n_paper_neg = 0
    n_venue_neg = 0
    n_coauthor_neg = 0
    n_map_by_others = 0

    pos_paper_thr = 0.8
    pos_coauthor_thr = pos_paper_thr

    pubs_overlap_thr = 0.5
    venues_overlap_thr = 0.05
    coauthors_overlap_thr = 0.05
    coauthors_overlap_thr_pos = 0.5
    thr_other = 0.2

    for i, name in enumerate(aminer_name_aid_to_pids):
        logger.info("name %d: %s", i, name)
        cur_name_dict = aminer_name_aid_to_pids[name]
        for aid in cur_name_dict:
            if aid not in aids_set:
                continue
            if aid not in aids_train:
                continue
            cur_pids = cur_name_dict[aid]
            cur_pids_raw = [int(x.split("-")[0]) for x in cur_pids]

            vids_1 = [paper_dict[str(x)].get("venue_id") for x in cur_pids_raw if
                      str(x) in paper_dict and "venue_id" in paper_dict[str(x)]]

            coauthors_1 = aminer_person_to_coauthors.get(aid, [])

            for pid in cur_pids:
                if pid in mag_name_pid_to_aid.get(name, {}):
                    aid_map = mag_name_pid_to_aid[name][pid]
                    pubs_mag_author = mag_name_aid_to_pids.get(name, {}).get(str(aid_map), [])

                    mag_person_pids = pubs_mag_author
                    common_pids = set(cur_pids) & set(mag_person_pids)
                    n_common_pubs = len(common_pids)
                    pubs_overlap_a = n_common_pubs / len(cur_pids)
                    pubs_overlap_m = n_common_pubs / len(mag_person_pids)

                    pubs_mag_author = [x.split("-")[0] for x in pubs_mag_author]
                    vids_2 = [paper_dict[str(x)].get("venue_id") for x in pubs_mag_author if
                              str(x) in paper_dict and "venue_id" in paper_dict[str(x)]]

                    coauthors_2 = mag_person_to_coauthors.get(str(aid_map), [])
                    coauthor_sim, c_sim1, c_sim2 = utils.top_coauthor_sim(coauthors_1, coauthors_2, topk=10)

                    # v_sim = utils.top_venue_sim(vids_1, vids_2, topk=None)

                    cur_key = aid + "~~~" + str(pid)
                    if cur_key in true_keys:
                        cur_truth = 1
                    elif cur_key in false_keys:
                        cur_truth = 0
                    else:
                        cur_truth = -1

                    if aid in aperson_to_mperson and int(aperson_to_mperson[aid]) == int(aid_map):
                        pos_triplets.append(
                            {"aid1": aid, "aid2": int(aid_map), "pid": pid, "name": name, "author_sim": pubs_overlap_a, "label": cur_truth})
                        continue
                    elif min(pubs_overlap_a, pubs_overlap_m) >= pos_paper_thr:
                        pos_triplets.append(
                            {"aid1": aid, "aid2": int(aid_map), "pid": pid, "name": name, "author_sim": pubs_overlap_a, "label": cur_truth})
                        continue
                    elif min(c_sim1, c_sim2) >= pos_coauthor_thr:
                        pos_triplets.append(
                            {"aid1": aid, "aid2": int(aid_map), "pid": pid, "name": name, "author_sim": coauthor_sim, "label": cur_truth})
                        continue

                    if False:
                        pass
                    elif pubs_overlap_a < pubs_overlap_thr and pubs_overlap_m < pubs_overlap_thr:
                        neg_triplets.append({"aid1": aid, "aid2": int(aid_map), "pid": pid, "name": name,
                                             "author_sim": min(pubs_overlap_a, pubs_overlap_m), "label": cur_truth})
                        n_paper_neg += 1
                        continue
                    # elif v_sim < venues_overlap_thr: # too few
                    #     neg_triplets.append({"aid1": aid, "aid2": int(aid_map), "pid": pid, "name": name,
                    #                          "author_sim": v_sim})
                    #     n_venue_neg += 1
                    elif coauthor_sim < coauthors_overlap_thr:
                        neg_triplets.append({"aid1": aid, "aid2": int(aid_map), "pid": pid, "name": name,
                                             "author_sim": coauthor_sim, "label": cur_truth})
                        n_coauthor_neg += 1
                        continue
                    elif aid in aperson_to_mperson and int(aperson_to_mperson[aid]) != int(aid_map):
                        if coauthor_sim < thr_other:
                            neg_triplets.append({"aid1": aid, "aid2": int(aid_map), "pid": pid, "name": name,
                                                 "author_sim": coauthor_sim, "label": cur_truth})
                            n_map_by_others += 1

    logger.info("n_paper_neg %d, n_venue_neg %d, n_coauthors_neg %d, n_map_by_others %d",
                n_paper_neg, n_venue_neg, n_coauthor_neg, n_map_by_others)
    cur_suffix = "posp" + str(pos_paper_thr) + "a" + str(pos_coauthor_thr) + "negp" + str(pubs_overlap_thr) + "a" + str(coauthors_overlap_thr) + "o" + str(thr_other)
    file_dir = join(settings.DATASET_DIR, "cs_ics")
    os.makedirs(file_dir, exist_ok=True)

    logger.info("number of postive triplets: %d", len(pos_triplets))
    utils.dump_json(pos_triplets, file_dir, "cs_triplets_via_author_sim_{}.json".format(cur_suffix))
    logger.info("number of negative triplets: %d", len(neg_triplets))
    utils.dump_json(neg_triplets, file_dir, "ics_triplets_via_author_sim_{}.json".format(cur_suffix))

    check_triplets_quality(suffix=cur_suffix)


def gen_kens_na_data():
    out_dir = join(settings.DATA_DIR, "kens-na-data")
    os.makedirs(out_dir, exist_ok=True)
    file_dir = settings.DATASET_DIR
    aminer_name_aid_to_mag_pids = utils.load_json(file_dir, "name_aid_to_pids_in_mid_filter.json")
    mag_name_aid_to_pids = utils.load_json(file_dir, "name_aid_to_pids_out_mid_filter.json")

    # paper_aid_to_mid_select = utils.load_json(file_dir, "paper_aid_to_mid_select.json")
    pubs_dict = utils.load_json(settings.DATASET_DIR, 'conna_pub_dict_mid.json')

    # pids_hit = set()
    # for aid in paper_aid_to_mid_select:
    #     pids_hit.add(str(paper_aid_to_mid_select[aid]))
    # for name in tqdm(aminer_name_aid_to_mag_pids):
    #     for aid in aminer_name_aid_to_mag_pids[name]:
    #         for pid in aminer_name_aid_to_mag_pids[name][aid]:
    #             pids_hit.add(pid.split("-")[0])
    #
    # for name in tqdm(mag_name_aid_to_pids):
    #     for aid in mag_name_aid_to_pids[name]:
    #         for pid in mag_name_aid_to_pids[name][aid]:
    #             pids_hit.add(pid.split("-")[0])
    pids_hit = set(pubs_dict.keys())

    pid_set = set()
    aid_set = set()
    name_set = set()
    for i, name in enumerate(aminer_name_aid_to_mag_pids):
        cur_name_dict = aminer_name_aid_to_mag_pids[name]
        name_set.add(name)
        for aid in cur_name_dict:
            aid_set.add(aid)
            pid_set |= {str(x) for x in cur_name_dict[aid]}

    pid_set = pids_hit

    pids_sorted = sorted(list(pid_set))
    aminer_aids_sorted = sorted(list(aid_set))
    logger.info("number of aminer authors %d", len(aminer_aids_sorted))
    aminer_entity_to_idx = {e: i for i, e in enumerate(pids_sorted + aminer_aids_sorted)}

    entity_dir = join(out_dir, "entity")
    os.makedirs(entity_dir, exist_ok=True)
    with open(join(entity_dir, "aminer.tsv"), "w") as wf:
        for pid in pids_sorted:
            wf.write(str(pid) + "\n")
        for aid in aminer_aids_sorted:
            wf.write(aid + "\n")
    logger.info("aminer entities written")

    mag_aid_set = set()
    for i, name in enumerate(mag_name_aid_to_pids):
        cur_name_dict = mag_name_aid_to_pids[name]
        for aid in cur_name_dict:
            mag_aid_set.add(str(aid))

    mag_aid_sorted = sorted(list(mag_aid_set))
    mag_entity_to_idx = {e: i for i, e in enumerate(pids_sorted + mag_aid_sorted)}
    with open(join(entity_dir, "mag.tsv"), "w") as wf:
        for pid in pids_sorted:
            wf.write(str(pid) + "\n")
        for aid in mag_aid_sorted:
            wf.write(str(aid) + "\n")
    logger.info("mag entities written")

    name_sorted = sorted(list(name_set))
    name_to_idx = {n: i for i, n in enumerate(name_sorted)}
    with open(join(out_dir, "relations.txt"), "w") as wf:
        for name in name_sorted:
            wf.write(name + "\n")
    logger.info("name relations written")

    # pairs_test = utils.load_json(file_dir, "eval_na_checking_pairs_conna_filter_test_2.json")
    pairs_test = utils.load_json(file_dir, "eval_na_checking_triplets_relabel1_test.json")
    # pairs_valid = utils.load_json(file_dir, "eval_na_checking_pairs_conna_filter_valid_2.json")
    pairs_valid = utils.load_json(file_dir, "eval_na_checking_triplets_relabel1_valid.json")

    aminer_kg_triplets_train = []
    aminer_kg_triplets_val = []
    aminer_kg_triplets_test = []
    for i, name in enumerate(aminer_name_aid_to_mag_pids):
        cur_name_dict = aminer_name_aid_to_mag_pids[name]
        cur_name_idx = name_to_idx[name]
        for aid in cur_name_dict:
            cur_aid_idx = aminer_entity_to_idx[str(aid)]
            cur_pids = cur_name_dict[aid]
            for pid in cur_pids:
                if str(pid.split("-")[0]) not in aminer_entity_to_idx:
                    continue
                cur_pid_idx = aminer_entity_to_idx[str(pid.split("-")[0])]
                aminer_kg_triplets_train.append([cur_pid_idx, cur_name_idx, cur_aid_idx])

    kg_dir = join(out_dir, "kg")
    os.makedirs(kg_dir, exist_ok=True)
    with open(join(kg_dir, "aminer-train.tsv"), "w") as wf:
        for item in aminer_kg_triplets_train:
            wf.write("\t".join([str(x) for x in item]) + "\n")
    logger.info("aminer train kg triplets written")

    mag_kg_triplets_train = []
    mag_kg_triplets_val = []
    mag_kg_triplets_test = []

    for pair in pairs_valid:
        aid1 = str(pair["aid1"])
        aid1_idx = aminer_entity_to_idx[aid1]
        aid2 = str(pair["aid2"])
        aid2_idx = mag_entity_to_idx[aid2]
        pid = str(pair["pid"].split("-")[0])
        pid_idx = aminer_entity_to_idx[pid]
        name = str(pair["name"])
        name_idx = name_to_idx[name]
        aminer_kg_triplets_val.append([pid_idx, name_idx, aid1_idx])
        mag_kg_triplets_val.append([pid_idx, name_idx, aid2_idx])

    with open(join(kg_dir, "aminer-val.tsv"), "w") as wf:
        for item in aminer_kg_triplets_val:
            wf.write("\t".join([str(x) for x in item]) + "\n")
    logger.info("aminer val kg triplets written")

    with open(join(kg_dir, "mag-val.tsv"), "w") as wf:
        for item in mag_kg_triplets_val:
            wf.write("\t".join([str(x) for x in item]) + "\n")
    logger.info("mag val kg triplets written")

    for pair in pairs_test:
        aid1 = str(pair["aid1"])
        aid1_idx = aminer_entity_to_idx[aid1]
        aid2 = str(pair["aid2"])
        aid2_idx = mag_entity_to_idx[aid2]
        # pid = str(pair["pid"])
        pid = str(pair["pid"].split("-")[0])
        pid_idx = aminer_entity_to_idx[pid]
        name = str(pair["name"])
        name_idx = name_to_idx[name]
        aminer_kg_triplets_test.append([pid_idx, name_idx, aid1_idx])
        mag_kg_triplets_test.append([pid_idx, name_idx, aid2_idx])

    with open(join(kg_dir, "aminer-test.tsv"), "w") as wf:
        for item in aminer_kg_triplets_test:
            wf.write("\t".join([str(x) for x in item]) + "\n")
    logger.info("aminer test kg triplets written")

    with open(join(kg_dir, "mag-test.tsv"), "w") as wf:
        for item in mag_kg_triplets_test:
            wf.write("\t".join([str(x) for x in item]) + "\n")
    logger.info("mag test kg triplets written")

    for i, name in enumerate(mag_name_aid_to_pids):
        cur_name_dict = mag_name_aid_to_pids[name]
        cur_name_idx = name_to_idx[name]
        for aid in cur_name_dict:
            cur_aid_idx = mag_entity_to_idx[str(aid)]
            cur_pids = cur_name_dict[aid]
            for pid in cur_pids:
                if str(pid.split("-")[0]) not in mag_entity_to_idx:
                    continue
                cur_pid_idx = mag_entity_to_idx[str(pid.split("-")[0])]
                mag_kg_triplets_train.append([cur_pid_idx, cur_name_idx, cur_aid_idx])

    with open(join(kg_dir, "mag-train.tsv"), "w") as wf:
        for item in mag_kg_triplets_train:
            wf.write("\t".join([str(x) for x in item]) + "\n")
    logger.info("mag train kg triplets written")

    seed_links_a2m = []
    seed_links_m2a = []
    for i in range(len(pids_sorted)):
        seed_links_a2m.append([i, i])
        seed_links_m2a.append([i, i])

    logger.info("number of pids %d", len(pids_sorted))

    oag_linking_dir = join(settings.DATA_DIR, "..", "oag-2-1")
    aperson_to_mperson = load_oag_linking_pairs(oag_linking_dir, "author_linking_pairs_2020.txt")
    for aid in aperson_to_mperson:
        if aid not in aminer_entity_to_idx:
            continue
        aid_idx = aminer_entity_to_idx[aid]
        mid_map = str(aperson_to_mperson[aid])
        if mid_map not in mag_entity_to_idx:
            continue
        mid_idx = mag_entity_to_idx[mid_map]
        seed_links_a2m.append([aid_idx, mid_idx])
        seed_links_m2a.append([mid_idx, aid_idx])

    links_dir = join(out_dir, "seed_alignlinks")
    os.makedirs(links_dir, exist_ok=True)
    with open(join(links_dir, "aminer-mag.tsv"), "w") as wf:
        for pair in seed_links_a2m:
            wf.write("\t".join([str(x) for x in pair]) + "\n")

    with open(join(links_dir, "mag-aminer.tsv"), "w") as wf:
        for pair in seed_links_m2a:
            wf.write("\t".join([str(x) for x in pair]) + "\n")
    logger.info("seed links written")


def gen_perfect_training_pairs():
    file_dir = settings.DATASET_DIR

    pairs_valid = utils.load_json(file_dir, "eval_na_checking_triplets_relabel1_valid.json")
    pairs_test = utils.load_json(file_dir, "eval_na_checking_triplets_relabel1_test.json")

    pos_pairs = []
    neg_pairs = []

    for item in tqdm(pairs_valid + pairs_test):
        item["aid"] = item["aid1"]
        if item["label"] == 1:
            pos_pairs.append(item)
        else:
            neg_pairs.append(item)

    neg_pairs_copy = deepcopy(neg_pairs)
    dup_times = len(pos_pairs) // len(neg_pairs_copy)
    logger.info("dup times %d", dup_times)
    for _ in range(1, dup_times):
        neg_pairs += neg_pairs_copy

    logger.info("n_pos pairs %d, n_neg pairs %d", len(pos_pairs), len(neg_pairs))

    pairs = pos_pairs + neg_pairs
    labels = [1] * len(pos_pairs) + [0] * len(neg_pairs)

    utils.dump_json(pairs, file_dir, "training_pairs_perfect.json")
    utils.joblib_dump_obj(labels, settings.OUT_DATASET_DIR, "training_pairs_perfect_labels.pkl")
    utils.dump_json(pos_pairs, file_dir, "positive_paper_author_pairs_conna_clean_fake.json")
    utils.dump_json(neg_pairs, file_dir, "negative_paper_author_pairs_conna_clean_fake.json")


def gen_perfect_pairs_input_mat():
    # pairs = utils.load_json(settings.DATASET_DIR, "training_pairs_perfect.json")
    pos_pairs = utils.load_json(settings.DATASET_DIR, "positive_paper_author_pairs_conna_clean_fake.json")
    neg_pairs = utils.load_json(settings.DATASET_DIR, "negative_paper_author_pairs_conna_clean_fake.json")
    labels = [1] * len(pos_pairs) + [0] * len(neg_pairs)
    pairs = pos_pairs + neg_pairs

    pair_to_mat_in = utils.joblib_load_obj(settings.OUT_DATASET_DIR, "pa_pair_to_mat_mid_in.pkl")

    attr_sim_mats = []
    empty_cnt = 0
    for pair in tqdm(pairs):
        pid = pair["pid"]
        aid1 = pair["aid"]
        # aid2 = pair["aid2"]

        cur_key_in = "{}~~~{}".format(aid1, pid)
        # cur_key_out = "{}~~~{}".format(aid2, pid)
        if len(pair_to_mat_in[cur_key_in]) == 0:
            empty_cnt += 1
        attr_sim_mats.append(pair_to_mat_in[cur_key_in])

    print("empty cnt", empty_cnt)
    assert len(pairs) == len(attr_sim_mats)

    print("number of mats", len(attr_sim_mats))
    utils.joblib_dump_obj(attr_sim_mats, settings.OUT_DATASET_DIR, "paper_author_matching_input_mat_perfect.pkl")
    utils.joblib_dump_obj(labels, settings.OUT_DATASET_DIR, "training_pairs_perfect_labels.pkl")


def gen_perfect_pairs_input_mat_kddcup():
    file_dir = settings.DATASET_DIR
    name_aid_to_pids = utils.load_json(file_dir, "train_author_pub_index_profile.json")
    pos_pairs = utils.load_json(settings.DATASET_DIR, "positive_paper_author_pairs_conna_clean_fake.json")
    neg_pairs = utils.load_json(settings.DATASET_DIR, "negative_paper_author_pairs_conna_clean_fake.json")
    paper_dict = utils.load_json(file_dir, "paper_dict_used_mag.json")  # str: dict

    emb_dir = "/home/zfj/research-data/conna/kddcup/emb-228/"
    word_emb_mat = utils.load_data(emb_dir, "word_emb.array")
    # author_emb_mat = utils.load_data(emb_dir, "author_emb.array")
    pub_feature_dict = utils.load_data(emb_dir, "pub_feature.ids")

    pairs = pos_pairs + neg_pairs
    labels = [1] * len(pos_pairs) + [0] * len(neg_pairs)

    attr_sim_mats = []
    valid_cnt = 0

    for i, pair in enumerate(pairs):
        if i % 100 == 0:
            logger.info("pair %d, valid cnt %d", i, valid_cnt)
            # if i > 0 and len(attr_sim_mats[-1]) > 0:
            #     print(attr_sim_mats_dot[-1][0])

        pid = pair["pid"]
        aid = pair["aid"]
        name = pair["name"]
        cur_author_pubs = name_aid_to_pids[name][aid]
        cur_pids = [x for x in cur_author_pubs if x != pid]
        cur_paper_year = paper_dict.get(str(pid.split("-")[0]), {}).get("year", 2022)  # this line works str(pid)
        # print("cur_pids_before", cur_pids)
        if len(cur_pids) <= 10:
            pids_selected = cur_pids
        else:
            papers_attr = [(x, paper_dict[str(x.split("-")[0])]) for x in cur_pids if str(x.split("-")[0]) in paper_dict]
            papers_sorted = sorted(papers_attr, key=lambda x: abs(cur_paper_year - x[1].get("year", 2022)))
            pids_selected = [x[0] for x in papers_sorted][:10]

        cur_mats = []
        flag = False

        if pid not in pub_feature_dict:
            attr_sim_mats.append(cur_mats)
            continue

        author_id_list, author_idf_list, word_id_list, word_idf_list = pub_feature_dict[pid]

        p_embs = word_emb_mat[word_id_list[: settings.MAX_MAT_SIZE]]

        if len(p_embs) == 0:
            attr_sim_mats.append(cur_mats)
            continue

        for ap in pids_selected:
            if ap not in pub_feature_dict:
                # cur_mats.append(np.zeros(shape=(settings.MAX_MAT_SIZE, settings.MAX_MAT_SIZE), dtype=np.float16))
                continue
            ap_author_ids, _, ap_word_ids, _ = pub_feature_dict[ap]
            ap_embs = word_emb_mat[ap_word_ids[: settings.MAX_MAT_SIZE]]

            if len(ap_embs) == 0:
                # cur_mats.append(
                #     np.zeros(shape=(settings.MAX_MAT_SIZE, settings.MAX_MAT_SIZE), dtype=np.float16))
                pass
            else:
                cur_sim = cosine_similarity(p_embs, ap_embs)
                cur_mats.append(cur_sim)
                flag = True

        if flag:
            valid_cnt += 1
        else:
            print(pid, aid, name, pids_selected)

        attr_sim_mats.append(cur_mats)

        if i >= settings.TEST_SIZE - 1:
            break

    assert len(attr_sim_mats) == len(labels)
    utils.joblib_dump_obj(attr_sim_mats, settings.OUT_DATASET_DIR, "paper_author_matching_input_mat_perfect.pkl")
    utils.joblib_dump_obj(labels, settings.OUT_DATASET_DIR, "training_pairs_perfect_labels.pkl")


def gen_perfect_data_training_labels():
    file_dir = settings.DATASET_DIR

    if settings.data_source == "kddcup":
        pos_pairs = utils.load_json(file_dir, "positive_paper_author_pairs_conna_clean_1.json")
        neg_pairs = utils.load_json(file_dir, "negative_paper_author_pairs_conna_clean_1.json")
    elif settings.data_source == "aminer":
        pos_pairs = utils.load_json(file_dir, "positive_paper_author_pairs_conna_mid.json")
        neg_pairs = utils.load_json(file_dir, "negative_paper_author_pairs_conna_mid.json")
    else:
        raise NotImplementedError

    pairs_valid = utils.load_json(file_dir, "eval_na_checking_triplets_relabel1_valid.json")
    pairs_test = utils.load_json(file_dir, "eval_na_checking_triplets_relabel1_test.json")
    err_keys = set()
    for item in pairs_valid + pairs_test:
        aid1 = item["aid1"]
        pid = item["pid"]
        cur_key = aid1 + "~~~" + str(pid)
        if item["label"] == 0:
            err_keys.add(cur_key)

    labels = []
    for item in pos_pairs:
        aid1 = item["aid"]
        pid = item["pid"]
        cur_key = aid1 + "~~~" + str(pid)
        if cur_key in err_keys:
            labels.append(0)
        else:
            labels.append(1)
    labels += [0] * len(neg_pairs)
    print("len labels", len(labels), "sum labels", sum(labels))

    utils.joblib_dump_obj(labels, settings.OUT_DATASET_DIR, "training_pairs_perfect_labels.pkl")


def gen_clean_training_pairs_fake():  # only run for kddcup now
    file_dir = settings.DATASET_DIR

    valid_triplets = utils.load_json(file_dir, "eval_na_checking_triplets_relabel1_valid.json")
    test_triplets = utils.load_json(file_dir, "eval_na_checking_triplets_relabel1_test.json")

    if settings.data_source == "kddcup":
        pos_pairs = utils.load_json(file_dir, "positive_paper_author_pairs_conna_clean_1.json")
        neg_pairs = utils.load_json(file_dir, "negative_paper_author_pairs_conna_clean_1.json")
    elif settings.data_source == "aminer":
        pos_pairs = utils.load_json(file_dir, "positive_paper_author_pairs_conna_mid.json")
        neg_pairs = utils.load_json(file_dir, "negative_paper_author_pairs_conna_mid.json")
    else:
        raise NotImplementedError

    pos_pairs_clean = []
    neg_pairs_clean = []
    pos_key_set = set()
    neg_key_set = set()
    for item in valid_triplets + test_triplets:
        cur_pair = {"name": item["name"], "aid": item["aid1"], "pid": item["pid"]}
        if item["label"] == 0:
            neg_key_set.add("{}~~~{}".format(item["aid1"], item["pid"]))
            neg_pairs_clean += [cur_pair] * 5
        else:
            pos_key_set.add("{}~~~{}".format(item["aid1"], item["pid"]))
            pos_pairs_clean.append(cur_pair)

    for item in pos_pairs:
        cur_key = "{}~~~{}".format(item["aid"], item["pid"])
        if cur_key not in pos_key_set and cur_key not in neg_key_set:
            pos_pairs_clean.append(item)

    for item in neg_pairs:
        cur_key = "{}~~~{}".format(item["aid"], item["pid"])
        if cur_key not in pos_key_set and cur_key not in neg_key_set:
            neg_pairs_clean.append(item)

    print(len(pos_pairs_clean), len(neg_pairs_clean))
    utils.dump_json(pos_pairs_clean, file_dir, "positive_paper_author_pairs_conna_clean_fake.json")
    utils.dump_json(neg_pairs_clean, file_dir, "negative_paper_author_pairs_conna_clean_fake.json")


def gen_per_person_performance_sorted_by_ics():
    aid_to_ics_score = dd(list)
    aid_to_name = {}
    aid_to_scores = dd(list)
    if settings.data_source == "aminer":
        name_aid_to_pids_in = utils.load_json(settings.DATASET_DIR, "name_aid_to_pids_in_mid_filter.json")
        name_aid_to_pids_out = utils.load_json(settings.DATASET_DIR, "name_aid_to_pids_out_mid_filter.json")
        # person_to_coauthors_in = utils.load_json(settings.DATASET_DIR, "person_coauthors_in_mid.json")
        # person_to_coauthors_out = utils.load_json(settings.DATASET_DIR, "person_coauthors_out_mid.json")
        cmp_thr = 0.6
        psl_thr = 1.5
    elif settings.data_source == "kddcup":
        # name_aid_to_pids_in = utils.load_json(settings.DATASET_DIR, "train_author_pub_index_profile_enrich1.json")
        name_aid_to_pids_in = utils.load_json(settings.DATASET_DIR, "train_author_pub_index_profile.json")
        name_aid_to_pids_out = utils.load_json(settings.DATASET_DIR, "aminer_name_aid_to_pids_with_idx.json")
        # name_aid_to_pids_out = utils.load_json(settings.DATASET_DIR, "aminer_name_aid_to_pids_with_idx_enrich1.json")
        # person_to_coauthors_in = utils.load_json(settings.DATASET_DIR, "mag_person_coauthors.json")
        # person_to_coauthors_out = utils.load_json(settings.DATASET_DIR, "aminer_person_coauthors.json")
        cmp_thr = 0.4
        psl_thr = 1.3
    else:
        raise NotImplementedError
    
    test_pairs = utils.load_json(settings.DATASET_DIR, "eval_na_checking_triplets_relabel1_test.json")
    for pair in tqdm(test_pairs):
        aid1 = pair["aid1"]
        aid2 = pair["aid2"]
        name = pair["name"]
        pids_1 = name_aid_to_pids_in[name].get(aid1, [])
        pids_2 = name_aid_to_pids_out[name].get(aid2, [])
        cur_sim, cur_sim_max = utils.paper_overlap_ratio(pids_1, pids_2)
        aid_to_ics_score[str(aid1)].append(cur_sim)
    
    files = [join(settings.RESULT_DIR, "self-ablation", "20220621", "results_per_name_seed_{}.json".format(42 + i)) for i in range(0, 5)]

    for f in files:
        with open(f) as rf:
            for line in rf:
                items = line.strip().split()
                aid = items[0]
                name_raw = ""
                for pt in items[1].split("_"):
                    name_raw += pt[0].upper()
                    name_raw += pt[1:]
                    name_raw += " "
                name_raw = name_raw[:-1]
                aid_to_name[aid] = name_raw
                aid_to_scores[aid].append([float(items[2]), float(items[3])])
    
    aid_to_scores_cmp = dd(list)
    aid_to_scores_psl = dd(list)

    files = [join(settings.RESULT_DIR, "join-train", "20220621", "results_per_name_self_ls_tune_cmp_cmp_{}_psl_1.0_seed_{}.json".format(cmp_thr, 42 + i)) for i in range(0, 5)]
    
    for f in files:
        with open(f) as rf:
            for line in rf:
                items = line.strip().split()
                aid = items[0]
                aid_to_scores_cmp[aid].append([float(items[2]), float(items[3])])


    files = [join(settings.RESULT_DIR, "join-train", "20220621", "results_per_name_self_ls_tune_psl_cmp_0.0_psl_{}_seed_{}.json".format(psl_thr, 42 + i)) for i in range(0, 5)]
    
    for f in files:
        with open(f) as rf:
            for line in rf:
                items = line.strip().split()
                aid = items[0]
                aid_to_scores_psl[aid].append([float(items[2]), float(items[3])])
    
    aid_to_ics_score_sorted = sorted(aid_to_ics_score.items(), key=lambda x: np.mean(x[1]))

    wf = open(join(settings.RESULT_DIR, "per_person_performance_sorted_by_ics.txt"), "w")
    for i in range(len(aid_to_ics_score_sorted)):
        cur_aid = aid_to_ics_score_sorted[i][0]
        if cur_aid not in aid_to_name:
            continue
        cur_name = aid_to_name[cur_aid]
        wf.write(cur_name + " & ")
        metrics_base = np.mean(np.array(aid_to_scores[cur_aid]), axis=0)
        wf.write("{:.2f} & {:.2f} & ".format(metrics_base[0] * 100, metrics_base[1] * 100))
        metrics_cmp = np.mean(np.array(aid_to_scores_cmp[cur_aid]), axis=0)
        wf.write("{:.2f} & {:.2f} & ".format(metrics_cmp[0] * 100, metrics_cmp[1] * 100))
        metrics_psl = np.mean(np.array(aid_to_scores_psl[cur_aid]), axis=0)
        wf.write("{:.2f} & {:.2f} \\\\ \n".format(metrics_psl[0] * 100, metrics_psl[1] * 100))
        wf.flush()
    wf.close()


if __name__ == "__main__":
    # dump_paper_id_to_raw_features()
    # emb_model = EmbeddingModel.Instance()
    # emb_model.train()
    # cal_feature_idf()
    # dump_paper_id_to_feature_token_ids()
    # gen_pa_pair_to_input_mat_dict()
    # gen_pa_pairs_to_input_mat_train()
    # gen_pa_pairs_eval_to_input_mat(role="valid")
    # gen_ics_input_mat_mid(role="valid")

    # gen_cs_and_ics_triplets()
    # gen_cs_and_ics_percentage()
    # gen_cs_and_ics_percentage_filter_uncommon()
    # divide_cs_and_ics_percentage()
    # divide_cs_and_ics_percentage_remove_ics_false()
    # gen_fuzzy_and_neg_triplets(suffixes=["pctpospmin03negpmax009"])
    # gen_fuzzy_neg_pa_pairs_input_mat(suffixes=["pctpospmin03negpmax009"])
    # gen_cs_and_ics_triplets_input_mat(suffix=["pctpos01neg001", "pctpos015neg003", "pctpos02neg005", "pctpos025neg007", "pctpos03neg009"])
    # gen_cs_and_ics_triplets_input_mat(suffix=["pctpos035neg011", "pctpos04neg013", "pctpos045neg015", "pctpos05neg017", "pctpos055neg019"])
    # gen_cs_and_ics_triplets_input_mat(suffix=["pctdiv05", "pctdiv06", "pctdiv07", "pctdiv08", "pctdiv09", "pctdiv095"])
    # gen_cs_and_ics_triplets_input_mat(suffix=["pctcspmin03icspmin001", "pctcspmin03icspmin003", "pctcspmin03icspmin005", "pctcspmin03icspmin007", "pctcspmin03icspmin009", "pctcspmin03icspmin013", "pctcspmin03icspmin017"])
    # gen_cs_and_ics_triplets_input_mat(suffix=["pospmin01negpmax012other0.1", "pospmin02negpmax012other0.1", "pospmin03negpmax012other0.1", "pospmin04negpmax012other0.1", "pospmin05negpmax012other0.1"])
    # gen_cs_and_ics_triplets_input_mat(suffix=["posp0.9a0.9negp0.5a0.05o0.2", "posp0.8a0.8negp0.5a0.05o0.2", "posp0.7a0.7negp0.5a0.05o0.2", "posp0.6a0.6negp0.5a0.05o0.2", "posp0.5a0.5negp0.5a0.05o0.2"])

    # gen_debug_data()
    # gen_cs_neg_training_pairs_input_mat()

    # gen_training_paper_author_triplets()
    # gen_cs_and_ics_triplets_input_mat(suffix=[None])
    # gen_cs_neg_training_pairs_portion()
    # tune_ics_thr_only(cs_thr=0.3)
    # test_ics_intuition_gen_triplets()
    # try_gen_perfect_cs_and_ics()
    # gen_eval_triplets_relabel_to_label(role="valid")
    # gen_aff_sim_ics_triplets(thr=0.05)
    # analyze_cs_ics_false_partition()
    # gen_eval_triplets_one_to_many(role="valid")
    # cal_train_eval_person_overlap()
    # get_paper_id_to_aminer_person_all(bs=5)
    # get_pid_err_non_hit_to_label()
    # gen_kddcup_more_ics_eval_data()
    # gen_aid_to_pid_err_one_to_many_all()
    # enrich_eval_triplets_kddcup()
    # enrich_name_aid_to_pids_kddcup()

    # gen_training_pairs_input_mat_kddcup()
    # gen_eval_pair_input_mat_kddcup(role="valid")
    # gen_extra_triplets_via_author_sim_kddcup()
    # gen_cs_ics_input_mat_kddcup()
    # gen_fuzzy_neg_pairs_kddcup()
    # gen_fuzzy_neg_input_mat_kddcup()

    # gen_ics_triplets_for_eval_whoiswho(suffix="pctpospmin03negpmax009")
    # gen_ics_input_mat_mid_whoiswho(role="valid", suffix="pctpospmin03negpmax009")
    # try_gen_extra_ics_triplets_for_whoiswho(thr=0.1)
    # gen_extra_triplets_via_author_sim_whoiswho_new_4()
    # gen_kens_na_data()

    # gen_perfect_training_pairs()
    # gen_perfect_pairs_input_mat()
    # gen_perfect_data_training_labels()
    # gen_clean_training_pairs_fake()
    # gen_perfect_pairs_input_mat_kddcup()

    gen_per_person_performance_sorted_by_ics()

    logger.info("done")
