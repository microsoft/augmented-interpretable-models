import numpy as np
import imodelsx.augtree.llm
from os.path import join, dirname, abspath
path_to_repo = dirname(dirname(abspath(__file__)))

if __name__ == '__main__':
    # should print 'cached!'
    out = imodelsx.augtree.llm.expand_keyword('bad', cache_dir=join(path_to_repo, 'results', 'gpt3_cache'), verbose=True)