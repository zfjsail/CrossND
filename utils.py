from os.path import join
import os
import json
import joblib
import pickle
from collections import defaultdict as dd
from collections import Counter
import nltk
from nltk.corpus import stopwords
import multiprocessing
from torch.utils.data.sampler import Sampler

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp

punct = set(u''':!),.:;?.]}¢'"、。〉》」』〕〗〞︰︱︳﹐､﹒
﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､￠
々‖•·ˇˉ―′’”([{£¥'"‵〈《「『〔〖（［｛￡￥〝︵︷︹︻
︽︿﹁﹃﹙﹛﹝（｛“‘_…/''')

stemmer = nltk.stem.PorterStemmer()
stopwords_list = stopwords.words('english')
stopwords_list.extend(['-', 'at'])
stopwords_list = set(stopwords_list)


def joblib_dump_obj(obj, wfdir, wfname):
    logger.info('dumping %s ...', wfname)
    joblib.dump(obj, join(wfdir, wfname))
    logger.info('%s dumped.', wfname)


def joblib_load_obj(rfdir, rfname):
    logger.info('loading %s ...', rfname)
    obj = joblib.load(join(rfdir, rfname))
    logger.info('%s loaded', rfname)
    return obj


def load_json(rfdir, rfname):
    logger.info('loading %s ...', rfname)
    with open(join(rfdir, rfname), 'r', encoding='utf-8') as rf:
        data = json.load(rf)
        logger.info('%s loaded', rfname)
        return data


def dump_json(obj, wfdir, wfname):
    logger.info('dumping %s ...', wfname)
    with open(join(wfdir, wfname), 'w', encoding='utf-8') as wf:
        json.dump(obj, wf, indent=4, ensure_ascii=False)
    logger.info('%s dumped.', wfname)


def serialize_embedding(embedding):
    return pickle.dumps(embedding)


def deserialize_embedding(s):
    return pickle.loads(s)


def dump_data(obj, wfpath, wfname):
    os.makedirs(wfpath, exist_ok=True)
    with open(os.path.join(wfpath, wfname), 'wb') as wf:
        pickle.dump(obj, wf)


def load_data(rfpath, rfname):
    with open(os.path.join(rfpath, rfname), 'rb') as rf:
        return pickle.load(rf)


def processed_by_multi_thread(function, arg_list):
    num_thread = int(multiprocessing.cpu_count()/2)
    pool = multiprocessing.Pool(num_thread)
    pool.map(function, arg_list)
    pool.close()
    pool.join()


def stem(word):
    return stemmer.stem(word)


def clean_name(name):
    if name is None:
        return ""
    x = [k.strip() for k in name.lower().strip().replace(".", " ").replace("-", " ").split()]
    return "_".join(x)


def clean_sentence(text, stemming=False):
    for token in punct:
        text = text.replace(token, "")
    words = text.split()
    processed_words = []
    for w in words:
        w = w.lower()
        if w in stopwords_list:
            continue
        if stemming:
            w = stem(w)
        processed_words.append(w)
    words = processed_words
    return words


def extract_common_features(item):
    title_features = clean_sentence(item["title"], stemming=True)
    keywords_features = []
    keywords = item.get("keywords")
    if keywords:
        for k in keywords:
            keywords_features.extend(clean_sentence(k, stemming=True))
    venue_features = []
    venue_name = item.get('venue', '')
    if len(venue_name) > 2:
        venue_features = clean_sentence(venue_name.lower(), stemming=True)
    return title_features, keywords_features, venue_features


def extract_author_features(item, order=None):
    title_features, keywords_features, venue_features = extract_common_features(item)
    word_features = title_features + keywords_features + venue_features
    author_features = []

    for i, author in enumerate(item["authors"]):
        if order is not None and i != order:
            continue
        org_name = clean_sentence(author.get("org", ""), stemming=True)
        if len(org_name) > 2:
            word_features += org_name

        for j, coauthor in enumerate(item["authors"]):
            if i == j:
                continue
            coauthor_name = coauthor.get("name", "")
            if coauthor_name is None:
                continue
            if len(coauthor_name.strip()) > 0:
                if len(coauthor_name.strip()) > 2:
                    author_features.append(clean_name(coauthor_name))
                else:
                    author_features.append(coauthor_name.lower())

    return author_features, word_features


def top_coauthor_sim(c1, c2, topk=10):
    if topk is not None:
        c1 = c1[: topk]
        c2 = c2[: topk]
    c1 = [(x["name"], x["n"]) for x in c1]
    c2 = [(x["name"], x["n"]) for x in c2]
    c1_new = Counter(dict(c1))
    c2_new = Counter(dict(c2))
    common_top_coauthors = c1_new & c2_new
    n_common_c = sum(common_top_coauthors.values())
    nc1 = sum(c1_new.values())
    nc2 = sum(c2_new.values())
    c_sim_1, c_sim_2 = 1, 1
    if nc1 > 0 and nc2 > 0:
        c_sim_1 = n_common_c / nc1
        c_sim_2 = n_common_c / nc2
        c_sim = max(c_sim_1, c_sim_2)
        # c_sim = 2 * n_common_c / (nc1 + nc2)
    else:
        c_sim = 1
    return c_sim, c_sim_1, c_sim_2


def paper_overlap_ratio(pids1, pids2):
    common_pids = set(pids1) & set(pids2)
    n_common_pubs = len(common_pids)
    pubs_overlap_a = n_common_pubs / len(pids1)
    pubs_overlap_m = n_common_pubs / len(pids2)
    return min(pubs_overlap_a, pubs_overlap_m), max(pubs_overlap_a, pubs_overlap_m)


def gen_char_grams(sent, n=4):
    return [sent[i: i+4] for i in range(len(sent)-n+1)]


def aff_sim_ngrams(aff1, aff2, n=4):
    if not aff1 or not aff2:
        return 0.
    aff1 = aff1.lower()
    aff2 = aff2.lower()
    ngrams_1 = gen_char_grams(aff1, n)
    ngrams_2 = gen_char_grams(aff2, n)
    ngram_dict = dd(int)
    for g in ngrams_1:
        ngram_dict[g] += 1
    common_cnt = 0
    for g in ngrams_2:
        if g in ngram_dict:
            common_cnt += 1
            ngram_dict[g] -= 1
            if ngram_dict[g] == 0:
                ngram_dict.pop(g)
    return common_cnt/(min(len(ngrams_1), len(ngrams_2)) + 1)


class Singleton:
    """
    A non-thread-safe helper class to ease implementing singletons.
    This should be used as a decorator -- not a metaclass -- to the
    class that should be a singleton.
    The decorated class can define one `__init__` function that
    takes only the `self` argument. Also, the decorated class cannot be
    inherited from. Other than that, there are no restrictions that apply
    to the decorated class.
    To get the singleton instance, use the `Instance` method. Trying
    to use `__call__` will result in a `TypeError` being raised.
    """

    def __init__(self, decorated):
        self._decorated = decorated

    def Instance(self):
        """
        Returns the singleton instance. Upon its first call, it creates a
        new instance of the decorated class and calls its `__init__` method.
        On all subsequent calls, the already created instance is returned.
        """
        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated()
            return self._instance

    def __call__(self):
        raise TypeError('Singletons must be accessed through `Instance()`.')

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)


class ChunkSampler(Sampler):
    """
    Samples elements sequentially from some offset.
    Arguments:
        num_samples: # of desired data points
        start: offset where we should start selecting from
    """

    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples
