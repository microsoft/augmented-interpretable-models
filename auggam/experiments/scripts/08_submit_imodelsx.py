from imodelsx import submit_utils
from os.path import dirname, join
import os.path

repo_dir = dirname(dirname(dirname(os.path.abspath(__file__))))


CUSTOM_CHECKPOINTS = {
    "financial_phrasebank": ["ahmedrachid/FinancialBERT-Sentiment-Analysis"],
    "sst2": ["textattack/bert-base-uncased-SST-2"],
    "emotion": ["nateraw/bert-base-uncased-emotion"],
    "rotten_tomatoes": ["textattack/bert-base-uncased-rotten_tomatoes"],
    "dbpedia_14": ["fabriceyhc/bert-base-uncased-dbpedia_14"],
    "ag_news": ["textattack/bert-base-uncased-ag-news"],
    "trec": ["aychang/bert-base-cased-trec-coarse"],
}


################# Spacy tokenizer #################
params_shared_dict = {
    'ngrams': [7], #, 5, 3],
    'use_simple_tokenizer': [0], #, 'simplified'],
}

SHARED_CHECKPOINTS = ['bert-base-uncased', 'tfidfvectorizer', 'gpt2', 'gpt2-xl', 'llama_7b', 'linear_finetune', 'tfidfvectorizer'] #, 'hkunlp/instructor-xl']
DATASETS = ['dbpedia_14'] #, 'ag_news', 'trec', 'financial_phrasebank', 'sst2', 'emotion', 'rotten_tomatoes']

params_coupled_dict = {
    ('dataset', 'checkpoint'): [
        (d, c)
        for d in DATASETS
        for c in ['hkunlp/instructor-xl'] # SHARED_CHECKPOINTS + CUSTOM_CHECKPOINTS.get(d, [])
    ]
}

################# Simple tokenizer #################
# params_shared_dict = {
#     "ngrams": [7],  # , 5, 3],
#     "use_simple_tokenizer": [1],  # , 'simplified'],
# }
# SHARED_CHECKPOINTS = ["bert-base-uncased", "gpt2"]
# DATASETS = ["trec", "ag_news", "dbpedia_14"]
# params_coupled_dict = {
#     ("dataset", "checkpoint"): [
#         (d, c)
#         for d in DATASETS
#         for c in SHARED_CHECKPOINTS + CUSTOM_CHECKPOINTS.get(d, [])
#     ]
# }


# Args list is a list of dictionaries
# If you want to do something special to remove some of these runs, can remove them before calling run_args_list
args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)
submit_utils.run_args_list(
    args_list,
    script_name=join(repo_dir, "experiments", "08_fit_imodelsx.py"),
    actually_run=True,
    # reverse=True,
    # n_cpus=1,
    gpu_ids=[1],
)
