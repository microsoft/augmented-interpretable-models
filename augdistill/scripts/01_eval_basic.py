from imodelsx import submit_utils
from os.path import dirname, join
import os.path
repo_dir = dirname(dirname(os.path.abspath(__file__)))

params_shared_dict = {
    'seed': [1],
    'save_dir': [join(repo_dir, 'results')],
    'use_cache': [1],
}

params_coupled_dict = {
    # ('checkpoint', 'embedding_string_prompt', 'use_next_token_distr_embedding'): [
    #     ('bert-base-uncased', None, 0),
    #     ('textattack/distilbert-base-uncased-rotten-tomatoes', None, 0),
    #     ('hkunlp/instructor-xl', 'instructor_sentiment', 0),
    # ],
    ('checkpoint', 'embedding_string_prompt', 'use_next_token_distr_embedding', 'zeroshot_strategy'): [

        (checkpoint, string_prompt, 1, zeroshot_strategy)
        # for checkpoint in ['gpt2', 'gpt2-xl', 'meta-llama/Llama-2-7b-hf', 'mistralai/Mistral-7B-v0.1']
        for checkpoint in ['mistralai/Mixtral-8x7B-v0.1', 'meta-llama/Llama-2-13b-hf']
        for string_prompt in ['synonym', 'movie_sentiment']
        for zeroshot_strategy in ['pos_class', 'difference']
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
    # gpu_ids=[0, 1, 2, 3],
    gpu_ids=[[0, 1], [2, 3]],
    # debug_mode=True,
)
