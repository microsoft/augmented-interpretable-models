from imodelsx import EmbGAMClassifier
import numpy as np
import fire
from embgam import data
import pickle as pkl
from os.path import join, dirname, abspath
path_to_repo = dirname(dirname(abspath(__file__)))



def run_dataset(dataset: str='financial_phrasebank'):
    checkpoint = 'hkunlp/instructor-xl'

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


    # subsample
    # subsample_frac = 0.01
    # n = len(dset_train)
    # dset_train = dset_train.select(np.random.choice(
    #     range(n), replace=False,
    #     size=int(n * subsample_frac)
    # ))    
    # n2 = len(dset_val)
    # dset_val = dset_val.select(np.random.choice(
    #     range(n2), replace=False,
    #     size=int(n2 * subsample_frac)
    # ))

    # fit model
    m = EmbGAMClassifier(
        checkpoint=checkpoint,
        all_ngrams=True,
        ngrams=7,
        random_state=42,
        instructor_prompt=INSTRUCTIONS[dataset],
    )
    m.fit(dset_train[dataset_key_text], dset_train['label'], verbose=True)

    # predict
    # preds = m.predict(dset_val[dataset_key_text])
    # print('acc_val', np.mean(preds == dset_val['label']))
    print('\n\t++++++++Caching++++++++\n')
    m.cache_linear_coefs(dset_val[dataset_key_text])
    print('\n\t++++++++Predicting++++++++\n')
    preds = m.predict(dset_val[dataset_key_text])
    acc_val = np.mean(preds == dset_val['label'])
    print(dataset, 'acc_val', acc_val)

    pkl.dump(m, open(join(path_to_repo, f'results/instructor_{dataset}_imodelsx.pkl'), 'wb'))
    pkl.dump({'acc_val': acc_val}, open(join(path_to_repo, f'results/instructor_acc_{dataset}_imodelsx.pkl'), 'wb'))

if __name__ == '__main__':    
    fire.Fire(run_dataset)
    # dsets = {
    #     'financial_phrasebank': CHECKPOINT,
    #     'sst2': CHECKPOINT,
    #     'emotion': CHECKPOINT, 
    #     'rotten_tomatoes': CHECKPOINT,
    # }


    # for dataset in dsets.keys():
    #     run_dataset(dataset)