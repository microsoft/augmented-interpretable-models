import itertools
import os
from os.path import dirname
import sys
import submit_utils
repo_dir = dirname(dirname(os.path.abspath(__file__)))
# python /home/chansingh/emb-gam/experimental/fmri/02_fit_decoding.py

PARAMS_COUPLED_DICT = {
    ('save_dir', 'subsample_frac'): [
        # ('/home/chansingh/mntv1/deep-fMRI/results/linear_models/oct12', -1),
        ('/home/chansingh/mntv1/deep-fMRI/results/linear_models/subsamp_oct12', 0.1),
    ],
}

##########################################
# params shared across everything (higher up things are looped over first)
##########################################
PARAMS_SHARED_DICT = {
    'save_dir': ['/home/chansingh/mntv1/deep-fMRI/results/linear_models/oct12'],
    # things to vary
    'dset': ['trec', 'emotion', 'rotten_tomatoes', 'tweet_eval', 'sst2'],
    'model': [
        'bert-10__ndel=4fmri', 'glove__ndel=4fmri', 'eng1000__ndel=4fmri',
        'glovevecs', 'eng1000vecs', 'bowvecs',
    ],

    # things to average over
    'seed': [1],
}

ks_final, param_combos_final = submit_utils.combine_param_dicts(
    PARAMS_SHARED_DICT, PARAMS_COUPLED_DICT)
submit_utils.run_dicts(
    ks_final, param_combos_final,
    script_name='02_fit_decoding.py',
    actually_run=True,
    shuffle=False,
    reverse=False,
)
