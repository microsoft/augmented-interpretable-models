from tqdm import tqdm
from os.path import join
from datasets import load_from_disk
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import pickle as pkl
import os
from os.path import join as oj
path_to_current_file = os.path.dirname(os.path.abspath(__file__))


def get_dataset_for_logistic(
    checkpoint: str, ngrams: int, all_ngrams: bool, norm: bool,
    dataset, data_dir, data_dir_full, tokenizer_ngrams,
    seed: int = 1,
    subsample: int = -1,
    dataset_key_text: str = 'text',
):
    """
    args.dataset_key_text: str, e.g. "sentence" for sst2
    """

    y_train = dataset['train']['label']
    y_val = dataset['validation']['label']

    # load embeddings
    if 'bert-base' in checkpoint or 'distilbert' in checkpoint or 'BERT' in checkpoint:
        if all_ngrams:
            try:
                data = pkl.load(open(oj(data_dir, 'data.pkl'), 'rb'))
            except Exception as e:
                # print("\tcouldn't find", , 'trying', data_dir_full)
                data = pkl.load(open(oj(data_dir_full, 'data.pkl'), 'rb'))

            X_train = data['X_train']
            X_val = data['X_val']
        else:
            try:
                reloaded_dataset = load_from_disk(data_dir)
            except Exception as e:
                # print("\tcouldn't find", data_dir, 'trying', data_dir_full)
                try:
                    reloaded_dataset = load_from_disk(data_dir_full)
                except Exception as e:
                    print("\tcouldn't find", data_dir, 'OR', data_dir_full)
                    print(e)
                    exit(1)

            X_train = np.array(reloaded_dataset['train']['embs']).squeeze()
            X_val = np.array(reloaded_dataset['validation']['embs']).squeeze()

        if norm:
            X_train = (X_train - data['mean']) / data['sigma']
            X_val = (X_val - data['mean']) / data['sigma']
    elif 'vectorizer' in checkpoint:
        if all_ngrams:
            lower_ngram = 1
        else:
            lower_ngram = ngrams
        if checkpoint == 'countvectorizer':
            vectorizer = CountVectorizer(
                tokenizer=tokenizer_ngrams, ngram_range=(lower_ngram, ngrams))
        elif checkpoint == 'tfidfvectorizer':
            vectorizer = TfidfVectorizer(
                tokenizer=tokenizer_ngrams, ngram_range=(lower_ngram, ngrams))
        # vectorizer.fit(dataset['train']['sentence'])
        X_train = vectorizer.fit_transform(dataset['train'][dataset_key_text])
        X_val = vectorizer.transform(dataset['validation'][dataset_key_text])
    elif 'wordvecs' in checkpoint:
        if 'glove' in checkpoint:
            # download using this script: https://github.com/csinva/fmri/blob/master/00_glove_prepare.py
            embs_np_dir = '/home/chansingh/nlp_utils/glove'
            print('loading glove model...')
            vocab = np.load(open(join(embs_np_dir, 'vocab_npa.npy'), 'rb'))
            vset = set(vocab)
            vocab = {w: i for i, w in enumerate(vocab)}
            embs = np.load(open(join(embs_np_dir, 'embs_npa.npy'), 'rb'))

            def get_glove_embs(X, vocab, vset, embs, D=300):
                word_lists = [
                    # [word.lower() for word in sequence.split()]
                    sequence.split()
                    for sequence in X
                ]
                X_embs = np.zeros((len(word_lists), D))
                for i, word_list in enumerate(tqdm(word_lists)):
                    num = 0
                    for word in word_list:
                        if word in vset:
                            X_embs[i] += embs[vocab[word]]
                        else:
                           X_embs[i] += embs[vocab['unk']] 
                    if num > 0:
                        X_embs[i] /= len(word_list)
                    # print('num', num)
                    # X_embs.append(np.mean(x_embs, axis=0))
                return X_embs
            print('extracting glove embs...')
            X_train = get_glove_embs(
                dataset['train'][dataset_key_text], vocab, vset, embs)
            X_val = get_glove_embs(
                dataset['validation'][dataset_key_text], vocab, vset, embs)

    if subsample > 0:
        rng = np.random.default_rng(seed)
        idxs_subsample = rng.choice(
            X_train.shape[0], size=subsample, replace=False)
        X_train = X_train[idxs_subsample]
        y_train = np.array(y_train)[idxs_subsample]
    return X_train, X_val, y_train, y_val


def fit_and_score_logistic(X_train, X_val, y_train, y_val, r, seed: int = 1):
    """Fit logistic model and return acc
    """
    # model
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    m = LogisticRegressionCV(random_state=seed, refit=False, cv=cv)
    m.fit(X_train, y_train)
    r['model'] = deepcopy(m)

    # performance
    r['acc_train'] = m.score(X_train, y_train)
    r['acc_val'] = m.score(X_val, y_val)
    return r


class Word2VecVectorizer:
    def __init__(self, model):
        print("Loading in word vectors...")
        self.word_vectors = model
        print("Finished loading in word vectors")

    def transform(self, data):
        # determine the dimensionality of vectors
        v = self.word_vectors.get_vector('king')
        self.D = v.shape[0]

        X = np.zeros((len(data), self.D))
        n = 0
        emptycount = 0
        for sentence in data:
            tokens = sentence.split()
            vecs = []
            m = 0
            for word in tokens:
                try:
                    # throws KeyError if word not found
                    vec = self.word_vectors.get_vector(word)
                    vecs.append(vec)
                    m += 1
                except KeyError:
                    pass
            if len(vecs) > 0:
                vecs = np.array(vecs)
                X[n] = vecs.mean(axis=0)
            else:
                emptycount += 1
            n += 1
        print("Numer of samples with no words found: %s / %s" %
              (emptycount, len(data)))
        return X
