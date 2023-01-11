import os
import pickle as pkl

import numpy as np
from imodelsx import EmbGAMClassifier

from preprocess import clean_headlines, sample_data


def generate_fixed_data(huffpost_data):
    # generate fixed data for 2012-2015
    fixed_data = []
    fixed_labels = []
    for year in [2012, 2013, 2014, 2015]:
        data, labels = sample_data(huffpost_data, year, in_dist=True, frac=1)
        fixed_data.extend(data)
        fixed_labels.extend(labels)
    return fixed_data, fixed_labels


if __name__ == "__main__":
    # load data
    with open(f"Data/huffpost.pkl", "rb") as f:
        huffpost_data = pkl.load(f)

    huffpost_data = clean_headlines(huffpost_data)
    fixed_data, fixed_labels = generate_fixed_data(huffpost_data)

    # train model
    for i in range(1, 10):
        print("Using ", i * 10, "% of points from 2016 in distribution for training...")
        # iterate over 4 random seeds
        for seed in [42, 192, 852, 5555]:
            print("Random seed=", seed)
            # randomly subsample frac
            np.random.seed(seed)
            var_data, var_labels = sample_data(
                huffpost_data, year=2016, in_dist=True, frac=i / 10
            )
            m = EmbGAMClassifier(
                all_ngrams=True,
                checkpoint="bert-base-uncased",
                ngrams=2,
                random_state=seed,
            )
            m.fit(fixed_data + var_data, fixed_labels + var_labels)

            # save model
            if not os.path.exists("models/expt1"):
                os.makedirs("models/expt1")
            pkl.dump(m, open(f"models/expt1/expt1_frac_{i}_seed_{seed}.pkl", "wb"))
