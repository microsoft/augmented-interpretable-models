from datetime import datetime
import warnings
import sklearn
import embgam.config as config
from datasets import load_from_disk
import embgam.linear
import embgam.data
import pandas as pd
from copy import deepcopy
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AutoModelForCausalLM, pipeline
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import pickle as pkl
import os
from os.path import join as oj
from spacy.lang.en import English
import argparse
from tqdm import tqdm
path_to_current_file = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    PARAMS_LIST = [
        # {
        #     'dataset': ['tweet_eval'],
        #     'checkpoint': ['philschmid/BERT-tweet-eval-emotion'], # 'unitary/toxic-bert'], #'philschmid/BERT-tweet-eval-emotion'], # toxigen model throws err
        # },        
        {
            'dataset': ['emotion'],
            'checkpoint': ['nateraw/bert-base-uncased-emotion'],
        },
        {
            'dataset': ['rotten_tomatoes'],
            'checkpoint': ['textattack/bert-base-uncased-rotten_tomatoes'],
        },
        {
            'dataset': ['financial_phrasebank'],
            'checkpoint': ['ahmedrachid/FinancialBERT-Sentiment-Analysis'],
        },
        {
            'dataset': ['sst2'],
            'checkpoint': ['textattack/bert-base-uncased-SST-2'],
        },
    ]
    r = defaultdict(list)
    for d in tqdm(PARAMS_LIST):
        dataset_name = d['dataset'][0]
        checkpoint = d['checkpoint'][0]
        print('dataset', dataset_name, 'checkpoint', checkpoint)

        # set up data
        dataset, dataset_key_text = embgam.data.process_data_and_args(dataset_name)
        pipe = pipeline('text-classification', model=checkpoint, device=0)
        preds_dicts = pipe(dataset['validation'][dataset_key_text])
        preds = np.array([d['label'] for d in preds_dicts])
        pred_ids = list(map(pipe.model.config.label2id.get, preds))
        acc = np.mean(np.array(pred_ids) == dataset['validation']['label'])

        # save
        r['dataset'].append(dataset_name)
        r['checkpoint'].append(checkpoint)
        r['acc'].append(acc)
        save_dir = oj(config.results_dir)
        os.makedirs(save_dir, exist_ok=True)
        df = pd.DataFrame(r)
        df.to_csv(oj(save_dir, 'baseline_accs.csv'))
        print(save_dir, '\n', r, '\n-------------------SUCCESS------------------------\n\n')
