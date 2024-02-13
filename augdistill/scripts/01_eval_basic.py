from imodelsx import submit_utils
from os.path import dirname, join
import os.path
repo_dir = dirname(dirname(os.path.abspath(__file__)))

# Showcasing different ways to sweep over arguments
# Can pass any empty dict for any of these to avoid sweeping

# List of values to sweep over (sweeps over all combinations of these)
params_shared_dict = {
    'seed': [1],
    'save_dir': [join(repo_dir, 'results')],
    # pass binary values with 0/1 instead of the ambiguous strings True/False
    'use_cache': [1],
}

# List of tuples to sweep over (these values are coupled, and swept over together)
# Note: this is a dictionary so you shouldn't have repeated keys
params_coupled_dict = {
    ('checkpoint', 'embedding_string_prompt', 'use_next_token_distr_embedding'): [
        ('bert-base-uncased', None, 0),
        ('textattack/distilbert-base-uncased-rotten-tomatoes', None, 0),
        ('hkunlp/instructor-xl', 'instructor_sentiment', 0),

        ('gpt2', 'synonym', 1),
        ('gpt2-xl', 'synonym', 1),
        ('meta-llama/Llama-2-7b-hf', 'synonym', 1),
        ('mistralai/Mistral-7B-v0.1', 'synonym', 1),

        ('gpt2', 'movie_sentiment', 1),
        ('gpt2-xl', 'movie_sentiment', 1),
        ('meta-llama/Llama-2-7b-hf', 'movie_sentiment', 1),
        ('mistralai/Mistral-7B-v0.1', 'movie_sentiment', 1),
    ],
}

# Args list is a list of dictionaries
# If you want to do something special to remove some of these runs, can remove them before calling run_args_list
args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)
submit_utils.run_args_list(
    args_list,
    script_name=join(repo_dir, 'experiments', '01_eval.py'),
    actually_run=True,
    gpu_ids=[0, 1, 2, 3],
)
