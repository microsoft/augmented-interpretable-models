import itertools
import os
import random

##########################################
# main setting finetuned archs
##########################################    
GLOBAL_PARAMS = {
    'ngrams': [1, 2, 3, 4, 5, 6, 7],    
    'layer': ['last_hidden_state_mean'], # 'pooler_output'
}

PARAMS_LIST = [
    {
        'dataset': ['sst2'],    
        'checkpoint': ['textattack/roberta-base-SST-2'],
    },
    {
        'dataset': ['emotion'],
        'checkpoint': ['bhadresh-savani/roberta-base-emotion'], #        
    },
    {
       'dataset': ['rotten_tomatoes'],
        'checkpoint': ['textattack/roberta-base-rotten-tomatoes'], #        
    },
    {
        'dataset': ['financial_phrasebank'],
        # 'checkpoint': ['ahmedrachid/FinancialBERT-Sentiment-Analysis'],
        'checkpoint': ['abhilash1910/financial_roberta'], #  note this match isn't perfect
    }
]
    
num = 0
for PARAMS in PARAMS_LIST:
    ks = list(PARAMS.keys())
    vals = [PARAMS[k] for k in ks]

    ks2 = list(GLOBAL_PARAMS.keys())
    vals += [GLOBAL_PARAMS[k] for k in ks2]
    ks += ks2

    param_combinations = list(itertools.product(*vals)) # list of tuples
    random.shuffle(param_combinations)

    for i in range(len(param_combinations)):
        print(f'-------------------\n\n{num} / {len(param_combinations) * len(PARAMS_LIST)}\n---------------------\n')
        param_str = 'python ../01_extract_embeddings.py '    
        for j, key in enumerate(ks):
            param_str += '--' + key + ' ' + str(param_combinations[i][j]) + ' '
        # s.run(param_str)
        # print(param_str)
        os.system(param_str)
        num += 1