import os
import pickle as pkl

import numpy as np
from imodelsx import EmbGAMClassifier

from preprocess import clean_headlines, sample_data


def generate_fixed_data(huffpost_data):
    # generate fixed data for 2012-2013
    fixed_data = []
    fixed_labels = []
    for year in [2012, 2013]:
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
    for ngrams in [2, 3]:
        m = EmbGAMClassifier(
            all_ngrams=True,
            checkpoint="bert-base-uncased",
            ngrams=ngrams,
            random_state=42,
        )
        m.fit(fixed_data, fixed_labels)

        # save model
        if not os.path.exists("models/expt3"):
            os.makedirs("models/expt3")
        pkl.dump(m, open(f"models/expt3/expt3_ngrams_{ngrams}_bert_uncased.pkl", "wb"))

        m = EmbGAMClassifier(
            all_ngrams=True,
            checkpoint="textattack/bert-base-uncased-ag-news",
            ngrams=ngrams,
            random_state=42,
        )
        m.fit(fixed_data, fixed_labels)

        # save model
        if not os.path.exists("models/expt3"):
            os.makedirs("models/expt3")
        pkl.dump(m, open(f"models/expt3/expt3_ngrams_{ngrams}_bert_agnews.pkl", "wb"))
