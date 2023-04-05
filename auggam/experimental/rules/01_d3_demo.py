import os
import random
import pickle as pkl
import imodelsx
import json
from typing import List
import tqdm


if __name__ == '__main__':
    distribution_pairs = json.load(open('benchmark.json'))
    all_h2score = []
    for i, d in enumerate(tqdm.tqdm(distribution_pairs)):
        print('examples', len(d['positive_samples']), len(d['negative_samples']))
        N = 10
        h2score = imodelsx.explain_d3(
            pos=d['positive_samples'][:N], 
            neg=d['negative_samples'][:N], 
            note=f'benchmark {i}; can be anything, for logging purpose only',
            num_steps=100,
            num_folds=2,
            batch_size=64,
        )
        all_h2score.append(h2score)
        pkl.dump(all_h2score, open('benchmark_h2score.pkl', 'wb'))