from imodelsx import AugGAMClassifier
import numpy as np

from auggam import data
import pickle as pkl

dsets = {
    'financial_phrasebank': 'ahmedrachid/FinancialBERT-Sentiment-Analysis',
    'sst2': 'textattack/bert-base-uncased-SST-2',
    'emotion': 'nateraw/bert-base-uncased-emotion',
    'rotten_tomatoes': 'textattack/bert-base-uncased-rotten_tomatoes',
}

for dataset in dsets.keys():
    checkpoint = dsets[dataset]

    # set up data
    dset, dataset_key_text = data.process_data_and_args(dataset)
    dset_train = dset['train']
    dset_val = dset['validation']

    # fit model
    m = AugGAMClassifier(
        checkpoint=checkpoint,
        all_ngrams=True,
        ngrams=4,
        random_state=42,
    )
    m.fit(dset_train[dataset_key_text], dset_train['label'])

    # predict
    preds = m.predict(dset_val[dataset_key_text])
    # print('acc_val', np.mean(preds == dset_val['label']))
    m.cache_linear_coefs(dset_val[dataset_key_text])
    preds = m.predict(dset_val[dataset_key_text])
    print('acc_val', np.mean(preds == dset_val['label']))

    pkl.dump(m, open(f'../results/4gram_{dataset}_imodelsx.pkl', 'wb'))

    # interpret
    try:
        print('Total ngram coefficients: ', len(m.coefs_dict_))
        print('Most positive ngrams')
        for k, v in sorted(m.coefs_dict_.items(), key=lambda item: item[1], reverse=True)[:8]:
            print('\t', k, round(v, 2))
        print('Most negative ngrams')
        for k, v in sorted(m.coefs_dict_.items(), key=lambda item: item[1])[:8]:
            print('\t', k, round(v, 2))
    except:
        pass