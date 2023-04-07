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
    'use_verbose': [0],
    'max_depth': [8],
    'subsample_frac': [0.01, 0.05, 0.1, 0.25, 0.5, 0.75],
    'model_name': ['llm_tree', 'decision_tree', 'id3'], #, 'hstree'], # llm_tree
}

# Ensemble sweep
params_coupled_dict = {
}


# Args list is a list of dictionaries
# If you want to do something special to remove some of these runs, can remove them before calling run_args_list
args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)
submit_utils.run_args_list(
    args_list,
    script_name=os.path.join(repo_dir, 'experiments', '01_train_model.py'),
    actually_run=True,
    shuffle=False,
    n_cpus=2,
    repeat_failed_jobs=True,
)
