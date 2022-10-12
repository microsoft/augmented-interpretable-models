import os
from os.path import dirname


def test_encoding():
    repo_dir = dirname(dirname(os.path.abspath(__file__)))
    exit_value=os.system('python ' + os.path.join(repo_dir, '01_fit_encoding.py --save_dir ~/.tmp --ndelays 1 --feature glove --nboots 1 --sessions 1 -story_override'))
    assert exit_value == 0, 'default pipeline passed'

if __name__ == '__main__':
    test_encoding()