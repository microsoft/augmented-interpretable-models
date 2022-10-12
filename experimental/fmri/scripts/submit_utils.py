from typing import Dict, List, Tuple

import itertools
import os
from os.path import dirname
from os.path import join as oj
import sys
import random

repo_dir = dirname(dirname(os.path.abspath(__file__)))

def combine_param_dicts(
    PARAMS_SHARED_DICT: Dict[str, List],
    PARAMS_COUPLED_DICT: Dict[Tuple[str], List[Tuple]]
):
    # shared
    ks_shared = list(PARAMS_SHARED_DICT.keys())
    vals_shared = [PARAMS_SHARED_DICT[k] for k in ks_shared]
    for val in vals_shared:
        assert isinstance(val, list), f"param val {val} must be type list, got type {type(val)}"
    param_tuples_list_shared = list(
        itertools.product(*vals_shared))

    # coupled
    ks_coupled = list(PARAMS_COUPLED_DICT.keys())
    vals_coupled = [PARAMS_COUPLED_DICT[k] for k in ks_coupled]
    param_tuples_list_coupled = list(
        itertools.product(*vals_coupled))
    param_tuples_list_coupled_flattened = [
        sum(x, ()) for x in param_tuples_list_coupled]

    # final
    ks_final = ks_shared + list(sum(ks_coupled, ()))

    param_combos_final = [shared + combo
                          for shared in param_tuples_list_shared
                          for combo in param_tuples_list_coupled_flattened]
    return ks_final, param_combos_final
  

def run_dicts(
        ks_final: List, param_combos_final: List,
        cmd_python: str ='python',
        script_name: str = '01_fit_encoding.py',
        actually_run: bool=True,
        shuffle: bool=False,
        reverse: bool=False,
    ):
    if shuffle:
        random.shuffle(param_combos_final)
    if reverse:
        param_combos_final = param_combos_final[::-1]

    for i in range(len(param_combos_final)):
        param_str = cmd_python + ' ' + \
            os.path.join(repo_dir, script_name + ' ')
        for j, key in enumerate(ks_final):
            param_val = param_combos_final[i][j]
            if isinstance(param_val, list):
               param_str += '--' + key + ' ' + ' '.join(param_val) + ' ' 
            else:
                param_str += '--' + key + ' ' + str(param_val) + ' '
        print(
            f'\n\n-------------------{i + 1}/{len(param_combos_final)}--------------------\n' + param_str)
        try:
            if actually_run:
                os.system(param_str)
        except Exception as e:
            print(e)
