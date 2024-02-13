from imodelsx import submit_utils
from os.path import dirname, join
import os.path
repo_dir = dirname(dirname(os.path.abspath(__file__)))

params_shared_dict = {
    'seed': [1],
    'save_dir': [join(repo_dir, 'results')],
    'use_cache': [1],
    'batch_size': [64, 16, 4],
    'renormalize_embs_strategy': ['StandardScaler', 'QuantileTransformer'],
    'ngrams': [3],
}

params_coupled_dict = {
    ('checkpoint', 'embedding_string_prompt', 'embedding_ngram_strategy'): [
        ('bert-base-uncased', None, 'mean'),
        ('textattack/distilbert-base-uncased-rotten-tomatoes', None, 'mean'),
        ('hkunlp/instructor-xl', 'instructor_sentiment', 'mean'),
    ],
    ('checkpoint', 'embedding_string_prompt', 'embedding_ngram_strategy', 'zeroshot_strategy'): [

        (checkpoint, string_prompt, embedding_ngram_strategy, zeroshot_strategy)
        # 'gpt2', 'mistralai/Mistral-7B-v0.1']
        for checkpoint in ['gpt2-xl', 'meta-llama/Llama-2-7b-hf']
        # require 2-gpus
        # for checkpoint in ['meta-llama/Llama-2-13b-hf']
        # require 4 gpus
        # for checkpoint in ['mistralai/Mixtral-8x7B-v0.1', 'meta-llama/Llama-2-70b-hf']
        for embedding_ngram_strategy in ['next_token_distr']
        for string_prompt in ['synonym', 'movie_sentiment']
        for zeroshot_strategy in ['pos_class']  # , 'difference']
    ],
}

args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)
submit_utils.run_args_list(
    args_list,
    script_name=join(repo_dir, 'experiments', '01_eval.py'),
    actually_run=True,
    gpu_ids=[0, 1, 2, 3],
    # gpu_ids=[[0, 1], [2, 3]],
    # gpu_ids=[[0, 1, 2, 3]],
    # debug_mode=True,
)
