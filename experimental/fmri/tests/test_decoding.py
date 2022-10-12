import os
from os.path import dirname


def test_decoding():
    repo_dir = dirname(dirname(os.path.abspath(__file__)))
    exit_value = os.system('python ' + os.path.join(repo_dir,
                           '02_fit_decoding.py --save_dir ~/.tmp --model eng1000vecs --subsample_frac 0.1'))
    assert exit_value == 0, 'default decoding failed'


if __name__ == '__main__':
    test_decoding()
