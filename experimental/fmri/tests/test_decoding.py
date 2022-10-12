import os
from os.path import dirname


def test_decoding():
    repo_dir = dirname(dirname(os.path.abspath(__file__)))
    cmd = 'python ' + \
        os.path.join(repo_dir,
                     '02_fit_decoding.py --save_dir ~/.tmp --model eng1000vecs --subsample_frac 0.05 --use_cache 0')
    print(cmd)
    exit_value = os.system(cmd)
    assert exit_value == 0, 'default decoding failed'


if __name__ == '__main__':
    test_decoding()