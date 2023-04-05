from imodelsx import EmbGAMClassifier
import numpy as np
import fire
import pandas as pd
from embgam import data
import pickle as pkl
from os.path import join, dirname, abspath
from spacy.lang.en import English
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
path_to_repo = dirname(dirname(abspath(__file__)))



def run_dataset(dataset: str='sst2', ngrams=2):
    # set up data
    dset, dataset_key_text = data.process_data_and_args(dataset)
    dset_train = dset['train']
    dset_val = dset['validation']

    INSTRUCTIONS = {
        'rotten_tomatoes': 'Represent the Review sentence for classifying emotion as positive or negative; Input:',
        'sst2': 'Represent the Review sentence for classifying emotion as positive or negative; Input:',
        'emotion': 'Represent the Tweet for classifying emotion as positive or negative; Input:', 
        'financial_phrasebank': 'Represent the Financial statement for classifying emotion as positive or negative; Input:', 
    }

    # fit model
    # tok_simp = English().tokenizer
    # tokenizer_func = lambda x: [str(x) for x in tok_simp(x)] 
    # v = CountVectorizer(tokenizer=tokenizer_func, ngram_range=(ngrams, ngrams))
    v = CountVectorizer(ngram_range=(ngrams, ngrams))
    v.fit(dset_train['sentence'])

    # countvec coefs
    mat = v.transform(dset_train['sentence'])
    m = LogisticRegressionCV()
    m.fit(mat, dset_train['label'])

    # predict
    # preds = m.predict(dset_val[dataset_key_text])
    # print('acc_val', np.mean(preds == dset_val['label']))
    # print('\n\t++++++++Caching++++++++\n')
    # m.cache_linear_coefs(dset_val[dataset_key_text])
    print('\n\t++++++++Predicting++++++++\n')
    mat_val = v.transform(dset_val['sentence'])
    preds = m.predict(mat_val)
    acc_val = np.mean(preds == dset_val['label'])
    print(dataset, 'acc_val', acc_val)

    pkl.dump({'model': m, 'vectorizer': v}, open(join(path_to_repo, f'results/bow_{dataset}.pkl'), 'wb'))

if __name__ == '__main__':    
    # fire.Fire(run_dataset)
    run_dataset()