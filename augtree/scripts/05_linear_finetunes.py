from imodelsx import submit_utils
from os.path import dirname
import os.path
repo_dir = dirname(dirname(os.path.abspath(__file__)))


# python /home/chansingh/llm-tree/experiments/01_train_model.py --use_cache 0 --subsample_frac 1 --model_name llm_tree --max_depth 3 --use_llm_refine 1
# python /home/chansingh/llm-tree/experiments/01_train_model.py --use_cache 0 --subsample_frac 1 --model_name llm_tree --max_depth 3 --use_llm_refine 1 --use_llm_ties 1
# python /home/chansingh/llm-tree/experiments/01_train_model.py --use_cache 0 --subsample_frac 1 --model_name llm_tree --max_depth 3 --use_llm_refine 1 --use_llm_ties 1 --use_llm_prompt_context 1

# Showcasing different ways to sweep over arguments
# Can pass any empty dict for any of these to avoid sweeping

# List of values to sweep over (sweeps over all combinations of these)
params_shared_dict = {
    'seed': [1],
    'save_dir': [os.path.join(repo_dir, 'results', 'feb11')],
    'dataset_name': ['rotten_tomatoes', 'emotion', 'financial_phrasebank', 'sst2'], # tweet_eval
    'use_verbose': [1],
    'model_name': ['linear_finetune'],
}

params_coupled_dict = {}

args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)
submit_utils.run_args_list(
    args_list,
    script_name=os.path.join(repo_dir, 'experiments', '01_train_model.py'),
    actually_run=True,
    shuffle=True,
    n_cpus=1,
    repeat_failed_jobs=True,
)
