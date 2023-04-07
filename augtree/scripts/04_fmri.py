from imodelsx import submit_utils
from os.path import dirname
import os.path
repo_dir = dirname(dirname(os.path.abspath(__file__)))


# List of values to sweep over (sweeps over all combinations of these)
params_shared_dict = {
    'seed': [1],
    # 'save_dir': [os.path.join(repo_dir, 'results', 'fmri_feb24')],
    'save_dir': [os.path.join(repo_dir, 'results', 'fmri_mar3')],
    'dataset_name': ['csinva/fmri_language_responses'],
    'use_verbose': [0],
    # 'label_name': [f'voxel_{i}' for i in range(50)],
    'label_name': [f'voxel_{i}' for i in range(150)],
}

params_coupled_dict = {
    # single-tree models
    ('model_name', 'max_depth'): [
        (model_name, max_depth)
        for max_depth in [2, 4, 6, 8]
        for model_name in ['llm_tree', 'decision_tree']
    ],

    # ensemble-tree models
    # ('n_estimators', 'model_name', 'max_depth'): [
    #     (20, model_name, 8)
    #     for model_name in ['llm_tree', 'decision_tree'] #, 'decision_tree']
    # ],
    
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
    n_cpus=4,
    repeat_failed_jobs=True,
)
