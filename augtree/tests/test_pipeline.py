import os
from os.path import dirname, join

# PYTHONPATH=/home/chansingh/llm-tree/experiments python /home/chansingh/llm-tree/experiments/01_train_model.py --use_cache 0 --subsample_frac 0.1 --refinement_strategy None
# PYTHONPATH=/home/chansingh/llm-tree/experiments python /home/chansingh/llm-tree/experiments/01_train_model.py --use_cache 0 --subsample_frac 0.1 --refinement_strategy 'llm' --use_stemming 1
# PYTHONPATH=/home/chansingh/llm-tree/experiments python /home/chansingh/llm-tree/experiments/01_train_model.py --use_cache 0 --subsample_frac 0.1 --refinement_strategy None --model_name c45
def test_small_pipeline():
    repo_dir = dirname(dirname(os.path.abspath(__file__)))
    prefix = f'PYTHONPATH={join(repo_dir, "experiments")}'
    cmd = prefix + ' python ' + \
        os.path.join(repo_dir, 'experiments',
                     '01_train_model.py --use_cache 0 --subsample_frac 0.1 --refinement_strategy None')
    print(cmd)
    exit_value = os.system(cmd)
    assert exit_value == 0, 'default pipeline passed'


if __name__ == '__main__':
    test_small_pipeline()
