import datasets
import numpy as np
import os
from os.path import join
from encoding.ridge_utils.SemanticModel import SemanticModel
from matplotlib import pyplot as plt
from typing import List
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.feature_extraction.text import CountVectorizer
from encoding.feature_spaces import em_data_dir, data_dir, results_dir
from collections import defaultdict
import pandas as pd
import pickle as pkl
from sklearn import metrics

def get_dsets(dataset):
    
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
    # dset = dset.select(np.random.choice(len(dset), size=300, replace=False))
    X = dset[dataset_key_text]
    y = dset[dataset_key_label]

    dset_test = get_dset()[val_dset_key]
    # dset_test = dset_test.select(np.random.choice(len(dset_test), size=300, replace=False))
    X_test = dset_test[dataset_key_text]
    y_test = dset_test[dataset_key_label]
    return X, y, X_test, y_test

def get_vecs(X: List[str]) -> np.ndarray:
    eng1000 = SemanticModel.load(join(em_data_dir, 'english1000sm.hf5'))
    # extract features
    X = [
        [word.encode('utf-8') for word in sentence.split(' ')]
        for sentence in X
    ]
    feats = eng1000.project_stims(X)
    return feats

def get_embs_fmri(X: List[str], save_dir_fmri, perc_threshold=98) -> np.ndarray:
    feats = get_vecs(X)
    weights_npz = np.load(join(save_dir_fmri, 'weights.npz'))
    corrs_val = np.load(join(save_dir_fmri, 'corrs.npz'))['arr_0']
    
    weights = weights_npz['arr_0']
    N_DELAYS = 4
    # pretty sure this is right, but might be switched...
    weights = weights.reshape(N_DELAYS, -1, feats.shape[-1]) 
    # delays for coefs are not stored next to each other!! (see cell 25 file:///Users/chandan/Downloads/speechmodeltutorial-master/SpeechModelTutorial%20-%20Pre-run.html)
    # weights = weights.reshape(-1, N_DELAYS, feats.shape[-1]) 
    weights = weights.mean(axis=0).squeeze() # mean over delays dimension...
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

if __name__ == '__main__':
    dsets = ['trec', 'emotion', 'rotten_tomatoes', 'tweet_eval', 'sst2']
    seed = 1
    perc_threshold_fmri = 98
    save_dir_fmri = join(results_dir, 'eng1000', 'UTS03')
    save_dir = '/home/chansingh/mntv1/deep-fMRI/results/linear_models'
    os.makedirs(save_dir, exist_ok=True)
    
    
    for dset in dsets:
        X, y, X_test, y_test = get_dsets(dset)
        print('shapes', len(X), len(X_test))
        
        r = defaultdict(list)
        for k in ['eng1000vecs', 'eng1000fmri', 'bow']:
            print('computing', k)
            if k == 'eng1000vecs':
                feats_train = get_vecs(X)
                feats_test = get_vecs(X_test)
            elif k == 'eng1000fmri':
                feats_train = get_embs_fmri(X, save_dir_fmri, perc_threshold=perc_threshold_fmri)
                feats_test = get_embs_fmri(X_test, save_dir_fmri, perc_threshold=perc_threshold_fmri) 
            elif k == 'bow':
                feats_train, feats_test = get_bow_vecs(X, X_test)

            m = LogisticRegressionCV(random_state=seed, cv=3)
            m.fit(feats_train, y)
            r['feats'].append(k)
            r['acc'].append(m.score(feats_test, y_test))
            # r['roc_auc'].append(metrics.roc_auc_score(y_test, m.predict(feats_test)))
            r['feats_dim'].append(feats_train.shape[1])
        df = pd.DataFrame.from_dict(r).set_index('feats')
        df.to_pickle(join(save_dir, dset + '.pkl'))
