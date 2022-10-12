import argparse
import datasets
import numpy as np
import os
from os.path import join
import logging
from transformers import pipeline
from ridge_utils.SemanticModel import SemanticModel
from matplotlib import pyplot as plt
from typing import List
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.feature_extraction.text import CountVectorizer
from feature_spaces import em_data_dir, data_dir, results_dir, nlp_utils_dir
import feature_spaces
from collections import defaultdict
import pandas as pd
import pickle as pkl
from sklearn import metrics
from copy import deepcopy


def get_dsets(dataset: str, seed: int = 1, subsample_frac: float = None):

    # select subsets
    def get_dset():
        return datasets.load_dataset(dataset)
    if dataset == 'tweet_eval':
        def get_dset():
            return datasets.load_dataset('tweet_eval', 'hate')

    # print(get_dset())

    # select data keys
    dataset_key_text = 'text'
    if dataset == 'sst2':
        dataset_key_text = 'sentence'
    dataset_key_label = 'label'
    if dataset == 'trec':
        dataset_key_label = 'label-coarse'
    val_dset_key = 'validation'
    if dataset == 'trec':
        val_dset_key = 'test'

    dset = get_dset()['train']
    if subsample_frac is not None:
        rng = np.random.default_rng(seed=seed)
        dset = dset.select(
            rng.choice(len(dset),
                       size=int(len(dset) * subsample_frac),
                       replace=False)
        )
    X = dset[dataset_key_text]
    y = dset[dataset_key_label]

    dset_test = get_dset()[val_dset_key]
    # dset_test = dset_test.select(np.random.choice(len(dset_test), size=300, replace=False))
    X_test = dset_test[dataset_key_text]
    y_test = dset_test[dataset_key_label]
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
                        model='bert-base-uncased', device=0)
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
    weights = weights.reshape(args.ndelays, -1, feats.shape[-1])
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


def fit_decoding(dset, model, args):
    np.random.seed(args.seed)
    r = defaultdict(list)

    # get feats
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

    # fit model
    logging.info('Fitting logistic...')
    m = LogisticRegressionCV(random_state=args.seed, cv=3)
    m.fit(feats_train, y)

    # save stuff
    r['dset'].append(dset)
    r['feats'].append(model)
    r['acc'].append(m.score(feats_test, y_test))
    # r['roc_auc'].append(metrics.roc_auc_score(y_test, m.predict(feats_test)))
    r['feats_dim'].append(feats_train.shape[1])
    r['coef_'].append(deepcopy(m))
    df = pd.DataFrame.from_dict(r).set_index('feats')
    df.to_pickle(join(args.save_dir, f'{dset}_{model}.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default='UTS03')
    parser.add_argument("--ndelays", type=int, default=4)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--perc_threshold_fmri', type=float, default=98)
    parser.add_argument("--save_dir", type=str,
                        default='/home/chansingh/mntv1/deep-fMRI/results/linear_models/oct12')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--dset', type=str, default=None,
                        choices=['trec', 'emotion', 'rotten_tomatoes', 'tweet_eval', 'sst2'])
    parser.add_argument('--subsample_frac', type=float,
                        default=None, help='fraction of data to use for training')
    args = parser.parse_args()

    # set up logging
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    # logging.basicConfig(level=logging.DEBUG)
    for k in sorted(vars(args)):
        logger.info('\t' + k + ' ' + str(vars(args)[k]))

    # select args
    if args.model:
        models = [args.model]
    else:
        models = [
            'glovevecs', 'eng1000vecs', 'bowvecs',
            'glovefmri', 'eng1000fmri',
            'bert-10__ndel=4fmri',
        ]  # , 'glovevecs', 'eng1000vecs', 'eng1000fmri', 'bow'] # 'glovefmri'
    if args.dset:
        dsets = args.dataset
    else:
        dsets = ['trec', 'emotion', 'rotten_tomatoes', 'tweet_eval', 'sst2']

    os.makedirs(args.save_dir, exist_ok=True)

    # loop and fit decoding
    for dset in dsets:
        X, y, X_test, y_test = get_dsets(
            dset, seed=args.seed, subsample_frac=args.subsample_frac)
        for model in models:
            logging.info('computing model using ' + model)

            fit_decoding(dset, model, args)
    logging.info('Succesfully completed!')
