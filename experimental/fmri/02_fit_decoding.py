import argparse
import datasets
import numpy as np
import os
from os.path import join
import logging
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from transformers import pipeline
from ridge_utils.SemanticModel import SemanticModel
from matplotlib import pyplot as plt
from typing import List
from sklearn.linear_model import RidgeCV, LogisticRegressionCV, LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from feature_spaces import em_data_dir, data_dir, results_dir, nlp_utils_dir
import feature_spaces
from collections import defaultdict
import pandas as pd
import pickle as pkl
from sklearn import metrics
from copy import deepcopy
import sys

from tqdm import tqdm


def get_dsets(dataset: str, seed: int = 1, subsample_frac: float = None):

    # select subsets
    def get_dset():
        return datasets.load_dataset(dataset)
    if dataset == 'tweet_eval':
        def get_dset():
            return datasets.load_dataset('tweet_eval', 'hate')
    elif dataset == 'moral_stories':
        def get_dset():
            return datasets.load_dataset(
                'demelin/moral_stories', 'cls-action+norm-minimal_pairs')

    # default keys
    dataset_key_text = 'text'
    dataset_key_label = 'label'
    val_dset_key = 'validation'

    # changes
    if dataset == 'sst2':
        dataset_key_text = 'sentence'
    elif dataset == 'poem_sentiment':
        dataset_key_text = 'verse_text'
    elif dataset == 'trec':
        dataset_key_label = 'coarse_label'  # 'label-coarse'
        val_dset_key = 'test'
    elif dataset == 'go_emotions':
        dataset_key_label = 'labels'

    dset = get_dset()['train']
    if subsample_frac is not None and subsample_frac > 0:
        rng = np.random.default_rng(seed=seed)
        dset = dset.select(
            rng.choice(len(dset),
                       size=int(len(dset) * subsample_frac),
                       replace=False)
        )

    def remove_multilabel(X: List[str], y: List[List]):
        idxs = np.array(pd.Series(y).apply(len) == 1)
        X = np.array(X)[idxs].tolist()
        y = np.array([yy[0] for yy in y])[idxs].tolist()
        return X, y

    def get_X(d, dataset_key_text, dataset):
        if not dataset == 'moral_stories':
            return d[dataset_key_text]
        if dataset == 'moral_stories':
            text = np.vstack((d['immoral_action'], d['moral_action'])).T
            texts = []
            idxs = np.array(d['label']).astype(int)
            for i, idx in enumerate(tqdm(idxs)):
                texts.append(d['norm'][i] + ' ' + text[i, idx])
        return texts

    X = get_X(dset, dataset_key_text, dataset)
    y = dset[dataset_key_label]

    dset_test = get_dset()[val_dset_key]
    # dset_test = dset_test.select(np.random.choice(len(dset_test), size=300, replace=False))
    X_test = get_X(dset_test, dataset_key_text, dataset)
    y_test = dset_test[dataset_key_label]

    if dataset == 'go_emotions':
        X, y = remove_multilabel(X, y)
        X_test, y_test = remove_multilabel(X_test, y_test)

    return X, y, X_test, y_test


def get_word_vecs(X: List[str], model='eng1000') -> np.ndarray:
    if 'eng1000' in model:
        sm = SemanticModel.load(join(em_data_dir, 'english1000sm.hf5'))
    elif 'glove' in model:
        sm = SemanticModel.load_np(join(nlp_utils_dir, 'glove'))
    # extract features
    X = [
        [word.encode('utf-8') for word in sentence.split(' ')]
        for sentence in X
    ]
    feats = sm.project_stims(X)
    return feats


def get_ngram_vecs(X: List[str], model='bert-3') -> np.ndarray:
    if model.lower().startswith('bert-'):
        pipe = pipeline("feature-extraction",
                        model='bert-base-uncased',
                        truncation=True,
                        device=0)
        ngram_size = int(model.split('-')[1].split('__')[0])
    return feature_spaces.get_embs_from_text(
        X, embedding_function=pipe, ngram_size=ngram_size)


def get_embs_fmri(X: List[str], model, save_dir_fmri, perc_threshold=98) -> np.ndarray:
    if model.lower().startswith('bert-'):
        feats = get_ngram_vecs(X, model=model)
    else:
        feats = get_word_vecs(X, model=model)
    weights_npz = np.load(join(save_dir_fmri, 'weights.npz'))
    corrs_val = np.load(join(save_dir_fmri, 'corrs.npz'))['arr_0']

    weights = weights_npz['arr_0']
    # pretty sure this is right, but might be switched...
    ndelays = int(model[model.index('ndel=') + len('ndel='):])
    weights = weights.reshape(ndelays, -1, feats.shape[-1])
    # delays for coefs are not stored next to each other!!
    # (see cell 25 file:///Users/chandan/Downloads/speechmodeltutorial-master/SpeechModelTutorial%20-%20Pre-run.html)
    # weights = weights.reshape(-1, N_DELAYS, feats.shape[-1])
    weights = weights.mean(axis=0).squeeze()  # mean over delays dimension...
    embs = feats @ weights.T

    # subselect repr
    perc = np.percentile(corrs_val, perc_threshold)
    idxs = (corrs_val > perc)
    # print('emb dim', idxs.sum(), 'val corr cutoff', perc)
    embs = embs[:, idxs]

    return embs


def get_bow_vecs(X: List[str], X_test: List[str]):
    trans = CountVectorizer().fit(X).transform
    return trans(X).todense(), trans(X_test).todense()


def get_hf_embs(X: List[str], X_test: List[str], model: str):
    pipe = pipeline("feature-extraction", model=model, device=0)

    def get_llm_embs(examples: List[str]):
        def get_emb(x):
            return {'emb': pipe(x['text'])}
        text = datasets.Dataset.from_dict({'text': examples})
        out_list = text.map(get_emb)['emb']
        # out_list is (batch_size, 1, (seq_len + 2), 768)

        # convert to np array by averaging over len (can't just convert this since seq lens vary)
        num_examples = len(out_list)
        dim_size = len(out_list[0][0][0])
        embs = np.zeros((num_examples, dim_size))
        for i in tqdm(range(num_examples)):
            embs[i] = np.mean(out_list[i], axis=1)  # avg over seq_len dim
        return embs

    logging.info('Training embs...')
    feats_train = get_llm_embs(X)
    logging.info('Testing embs...')
    feats_test = get_llm_embs(X_test)
    return feats_train, feats_test


def get_feats(model: str, X: List[str], X_test: List[str], args):
    logging.info('Extracting features for ' + model)
    mod = model.replace('fmri', '').replace('vecs', '')
    if model.endswith('fmri'):
        save_dir_fmri = join(
            results_dir, 'encoding', mod, args.subject)
        feats_train = get_embs_fmri(
            X, mod, save_dir_fmri, perc_threshold=args.perc_threshold_fmri)
        feats_test = get_embs_fmri(
            X_test, mod, save_dir_fmri, perc_threshold=args.perc_threshold_fmri)
    elif model.endswith('vecs'):
        assert mod in ['bow', 'eng1000', 'glove']
        if mod == 'bow':
            feats_train, feats_test = get_bow_vecs(X, X_test)
        elif mod in ['eng1000', 'glove']:
            feats_train = get_word_vecs(X, model=mod)
            feats_test = get_word_vecs(X_test, model=mod)
    else:  # HF checkpoint
        feats_train, feats_test = get_hf_embs(X, X_test, model=mod)
    return feats_train, feats_test


def fit_decoding(
        model, feats_train, y_train, feats_test, y_test,
        fname_save, args, frac_train_to_drop=0.1):
    """Randomly fits to only 90% of training data
    """
    np.random.seed(args.seed)
    r = defaultdict(list)

    # fit model
    logging.info('Fitting logistic...')

    m = LogisticRegressionCV(random_state=args.seed, cv=3)
    if args.subsample_frac is None or args.subsample_frac < 0:
        feats_train, feats_drop, y_train, y_drop = train_test_split(
            feats_train, y_train, test_size=frac_train_to_drop, random_state=args.seed)
    m.fit(feats_train, y_train)

    # cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)
    # m = LogisticRegressionCV(random_state=args.seed, refit=True, cv=cv) # with refit, should get better performance but no variance
    # param_grid = {'C': np.logspace(-4, 4, 10)}
    # logistic = LogisticRegression(random_state=args.seed)

    # we should do this eventually to get the best performance
    # m = GridSearchCV(logistic, param_grid, refit=False, cv=cv)
    # m.fit(feats_train, y_train)

    # save stuff
    acc = m.score(feats_test, y_test)
    logging.info(f'acc {acc:0.2f}')
    r['dset'].append(args.dset)
    r['feats'].append(model)
    r['acc'].append(acc)
    # r['roc_auc'].append(metrics.roc_auc_score(y_test, m.predict(feats_test)))
    r['feats_dim'].append(feats_train.shape[1])
    r['coef_'].append(deepcopy(m))
    r['seed'].append(args.seed)
    df = pd.DataFrame.from_dict(r).set_index('feats')
    df.to_pickle(fname_save)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default='UTS03')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--perc_threshold_fmri', type=float, default=98)
    parser.add_argument("--save_dir", type=str, default='/home/chansingh/.tmp')
    parser.add_argument('--model', type=str, default='glovevecs',
                        help='Which model to extract features with. \
                            Ending in fmri uses model finetuned on fMRI. \
                            Ending in vecs uses a word-vector model. \
                            Otherwise uses HF checkpoint.')  # glovevecs, bert-10__ndel=4fmri, bert-base-uncased
    parser.add_argument('--dset', type=str, default='rotten_tomatoes',
                        choices=['trec', 'emotion', 'rotten_tomatoes', 'tweet_eval',
                                 'sst2', 'go_emotions', 'poem_sentiment', 'moral_stories'])
    # could also support more tweet dsets (currently only hate), imdb, ...
    parser.add_argument('--subsample_frac', type=float,
                        default=None, help='fraction of data to use for training. If none or negative, use all the data')
    parser.add_argument('--use_cache', type=int,
                        default=True, help='whether to use cache')
    args = parser.parse_args()

    # set up logging
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    # logging.basicConfig(level=logging.DEBUG)
    for k in sorted(vars(args)):
        logger.info('\t' + k + ' ' + str(vars(args)[k]))

    # check for caching
    fname_save = join(
        args.save_dir, f'{args.dset}_{args.model}_seed={args.seed}.pkl')
    if os.path.exists(fname_save) and args.use_cache:
        logging.info('\nAlready ran ' + fname_save + '!')
        logging.info('Skipping :)!\n')
        sys.exit(0)

    # get data
    X_train, y_train, X_test, y_test = get_dsets(
        args.dset, seed=args.seed, subsample_frac=args.subsample_frac)

    # fit decoding
    os.makedirs(args.save_dir, exist_ok=True)
    feats_train, feats_test = get_feats(args.model, X_train, X_test, args)
    fit_decoding(args.model, feats_train, y_train,
                 feats_test, y_test, fname_save, args)
    logging.info('Succesfully completed!')
