from imodelsx import EmbGAMClassifier
from preprocess import clean_headlines, sample_data
import numpy as np
import pickle as pkl
import os


if __name__ == "__main__":
    # load data
    with open(f"Data/huffpost.pkl", "rb") as f:
        huffpost_data = pkl.load(f)

    huffpost_data = clean_headlines(huffpost_data)

    # train model
    for train_year in [2012, 2013, 2014, 2015, 2016, 2017, 2018]:
        train_data, train_labels = sample_data(
            huffpost_data,
            year=train_year,
            in_dist=True,
            frac=1,
        )
        for test_year in [2012, 2013, 2014, 2015, 2016, 2017, 2018]:
            if train_year == test_year:
                continue

            print("Train year: ", train_year, "Test year:", test_year)
            # iterate over 4 random seeds
            for seed in [42, 192, 852, 5555]:
                print("Random seed=", seed)
                # randomly subsample frac
                np.random.seed(seed)
                var_data, var_labels = sample_data(
                    huffpost_data,
                    year=test_year,
                    in_dist=True,
                    frac=0.2,
                )
                m = EmbGAMClassifier(
                    all_ngrams=True,
                    checkpoint="bert-base-uncased",
                    ngrams=2,
                    random_state=seed,
                )
                m.fit(train_data + var_data, train_labels + var_labels)

                # save model
                if not os.path.exists("models/expt2"):
                    os.makedirs("models/expt2")
                pkl.dump(
                    m,
                    open(
                        f"models/expt2/expt2_train_{train_year}_test_{test_year}_seed_{seed}.pkl",
                        "wb",
                    ),
                )
